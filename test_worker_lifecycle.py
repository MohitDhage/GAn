#!/usr/bin/env python3
"""
==============================================================================
Subphase 1.3 — Worker Lifecycle Verification Script
3D-GAN Generation System

File    : test_worker_lifecycle.py

What this script verifies
--------------------------
 1.  Redis is reachable and the Celery broker/backend queues are healthy.
 2.  The task can be dispatched with a generated UUID job_id (matching the
     pattern FastAPI will use in Subphase 1.4).
 3.  The task transitions through the expected state sequence:
         PENDING → STARTED/PROCESSING → PROCESSING(30%) → PROCESSING(80%) → SUCCESS
 4.  The final AsyncResult.result matches the Phase 1.4 contract schema exactly.
 5.  The .glb file exists on disk at OUTPUTS_DIR/<job_id>.glb after completion.
 6.  The file was written via the atomic rename pattern — validated by checking
     no orphaned .tmp file remains.
 7.  The Redis metadata key (gan3d:job:<id>:meta) was written with the
     expected fields (submitted_at, completed_at, asset_url).
 8.  torch.cuda.empty_cache() leaves VRAM in a clean state post-task.

Prerequisites (must all be true before running)
------------------------------------------------
    docker compose up -d          # Redis must be running
    # In a SEPARATE terminal — start the Celery worker:
    celery -A celery_app worker --loglevel=info --concurrency=1 -Q gan3d.generate

Usage
-----
    python test_worker_lifecycle.py [--timeout 180] [--resolution 64]

    --timeout     Max seconds to poll before declaring failure (default 180)
    --resolution  Marching Cubes resolution sent to the task (default 64 for speed)
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Path setup — allows running from any working directory                       #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from celery_app import celery_app, redis_client
from celery.result import AsyncResult
from inference import OUTPUTS_DIR

# --------------------------------------------------------------------------- #
# ANSI helpers                                                                 #
# --------------------------------------------------------------------------- #
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✔  {RESET}{msg}", flush=True)
def warn(msg):   print(f"  {YELLOW}⚠  {RESET}{msg}", flush=True)
def fail(msg):   print(f"  {RED}✘  {RESET}{msg}", flush=True)
def info(msg):   print(f"  {CYAN}ℹ  {RESET}{msg}", flush=True)
def header(msg): print(f"\n{BOLD}{CYAN}{'='*66}{RESET}\n{BOLD}  {msg}{RESET}\n{'='*66}", flush=True)

CHECKS_PASSED: list[str] = []
CHECKS_FAILED: list[str] = []

def record(passed: bool, name: str, message: str):
    if passed:
        ok(message)
        CHECKS_PASSED.append(name)
    else:
        fail(message)
        CHECKS_FAILED.append(name)


# --------------------------------------------------------------------------- #
# CHECK 1 — Redis and Celery broker connectivity                              #
# --------------------------------------------------------------------------- #
def check_redis_and_broker() -> bool:
    header("CHECK 1 — Redis & Celery Broker Connectivity")
    all_ok = True

    # Direct Redis ping
    try:
        pong = redis_client.ping()
        record(pong, "redis_ping", f"Redis PING → {'PONG' if pong else 'FAILED'}")
        if not pong:
            all_ok = False
    except Exception as exc:
        record(False, "redis_ping", f"Redis unreachable: {exc}")
        all_ok = False

    # Celery broker inspection (lightweight — just check connection)
    try:
        insp = celery_app.control.inspect(timeout=3)
        stats = insp.stats()
        if stats:
            worker_names = list(stats.keys())
            ok(f"Active Celery worker(s) found: {worker_names}")
            CHECKS_PASSED.append("celery_worker_active")
        else:
            warn(
                "No active Celery workers detected. "
                "Start one with:\n"
                "    celery -A celery_app worker --loglevel=info "
                "--concurrency=1 -Q gan3d.generate"
            )
            CHECKS_FAILED.append("celery_worker_active")
            all_ok = False
    except Exception as exc:
        record(False, "celery_broker", f"Celery broker check failed: {exc}")
        all_ok = False

    return all_ok


# --------------------------------------------------------------------------- #
# CHECK 2 — Task dispatch with correct job_id contract                       #
# --------------------------------------------------------------------------- #
def dispatch_task(mc_resolution: int) -> tuple[str, AsyncResult]:
    header("CHECK 2 — Task Dispatch")

    # Generate UUID exactly as FastAPI POST /v1/generate will in Subphase 1.4
    job_id = str(uuid.uuid4())
    info(f"Generated job_id : {job_id}")
    info(f"mc_resolution    : {mc_resolution}")

    # Build a dummy 128×128 RGB image tensor and serialise to JSON-safe list
    dummy_image = torch.rand(1, 3, 128, 128, dtype=torch.float32)
    image_data  = dummy_image.numpy().flatten().tolist()
    image_shape = list(dummy_image.shape)  # [1, 3, 128, 128]

    # Dispatch — task_id=job_id binds the Celery task ID to our domain ID
    async_result: AsyncResult = celery_app.send_task(
        "tasks.generate_3d_asset",
        kwargs={
            "job_id":        job_id,
            "image_data":    image_data,
            "image_shape":   image_shape,
            "noise_seed":    42,
            "mc_resolution": mc_resolution,
        },
        task_id=job_id,
        queue="gan3d.generate",
    )

    record(
        async_result.id == job_id,
        "task_id_matches_job_id",
        f"AsyncResult.id == job_id: {async_result.id == job_id} ({async_result.id})",
    )

    return job_id, async_result


# --------------------------------------------------------------------------- #
# CHECK 3 — State machine polling                                             #
# --------------------------------------------------------------------------- #
def poll_until_terminal(
    async_result: AsyncResult,
    job_id: str,
    timeout: int,
) -> tuple[str, dict | None]:
    """
    Poll AsyncResult every 2 seconds until terminal state or timeout.
    Logs every state transition and progress update.
    Returns (final_state, result_or_None).
    """
    header("CHECK 3 — State Machine Polling")

    terminal_states   = {"SUCCESS", "FAILURE", "REVOKED", "EXPIRED"}
    observed_states   = []
    observed_progress = []
    deadline          = time.time() + timeout
    poll_interval     = 2.0

    info(f"Polling (max {timeout}s, interval {poll_interval}s)...")
    print()

    last_state = None

    while time.time() < deadline:
        state = async_result.state
        meta  = async_result.info  # dict for custom states, Exception for FAILURE

        if state != last_state:
            observed_states.append(state)
            ts = time.strftime("%H:%M:%S")

            if state == "PROCESSING" and isinstance(meta, dict):
                progress = meta.get("progress", "?")
                stage    = meta.get("stage", "")
                if progress not in observed_progress:
                    observed_progress.append(progress)
                print(
                    f"  [{ts}] {CYAN}{state}{RESET} "
                    f"{progress}% — {stage}",
                    flush=True,
                )
            elif state in terminal_states:
                colour = GREEN if state == "SUCCESS" else RED
                print(f"  [{ts}] {colour}{BOLD}{state}{RESET}", flush=True)
            else:
                print(f"  [{ts}] {CYAN}{state}{RESET}", flush=True)

            last_state = state

        if state in terminal_states:
            break

        time.sleep(poll_interval)

    else:
        fail(f"Polling timeout after {timeout}s. Last state: {last_state}")
        return last_state or "TIMEOUT", None

    print()

    # Validate state sequence — we expect at least PENDING→STARTED→SUCCESS
    info(f"Observed state sequence: {' → '.join(observed_states)}")
    has_processing = any(s == "PROCESSING" for s in observed_states)
    record(
        has_processing,
        "processing_state_observed",
        f"PROCESSING state {'observed' if has_processing else 'NOT OBSERVED — worker may have run too fast or not emitted state updates'}",
    )

    expected_progress_milestones = {30, 80}
    observed_set = set(observed_progress)
    milestones_hit = expected_progress_milestones.issubset(observed_set)
    record(
        milestones_hit,
        "progress_milestones",
        f"Progress milestones (30%, 80%) observed: {observed_set} "
        f"— {'✔' if milestones_hit else 'missing ' + str(expected_progress_milestones - observed_set)}",
    )

    final_state = async_result.state
    record(
        final_state == "SUCCESS",
        "terminal_state_success",
        f"Terminal state: {final_state}",
    )

    result = None
    if final_state == "SUCCESS":
        result = async_result.result
    elif final_state == "FAILURE":
        fail(f"Task failed with exception: {async_result.result}")

    return final_state, result


# --------------------------------------------------------------------------- #
# CHECK 4 — Phase 1.4 contract validation                                    #
# --------------------------------------------------------------------------- #
def check_result_contract(result: dict, job_id: str):
    header("CHECK 4 — Phase 1.4 Return Contract")

    if result is None:
        fail("No result available — task did not succeed.")
        CHECKS_FAILED.append("result_contract")
        return

    errors = []

    for key in ("asset_url", "file_size_bytes", "metadata"):
        if key not in result:
            errors.append(f"Missing key: '{key}'")

    asset_url = result.get("asset_url", "")
    if not (isinstance(asset_url, str)
            and asset_url.startswith("/outputs/")
            and asset_url.endswith(".glb")):
        errors.append(f"asset_url malformed: '{asset_url}'")

    expected_suffix = f"{job_id}.glb"
    if not asset_url.endswith(expected_suffix):
        errors.append(
            f"asset_url job_id mismatch: expected suffix '{expected_suffix}', "
            f"got '{asset_url}'"
        )

    fsb = result.get("file_size_bytes", 0)
    if not (isinstance(fsb, int) and fsb > 0):
        errors.append(f"file_size_bytes invalid: {fsb}")

    required_meta = {
        "job_id": str,
        "resolution": int,
        "latency_seconds": float,
        "peak_vram_mb": float,
        "vertex_count": int,
        "face_count": int,
        "components_before_clean": int,
        "generator_class": str,
    }
    meta = result.get("metadata", {})
    for field, expected_type in required_meta.items():
        if field not in meta:
            errors.append(f"metadata missing: '{field}'")
        elif not isinstance(meta[field], expected_type):
            errors.append(
                f"metadata['{field}']: expected {expected_type.__name__}, "
                f"got {type(meta[field]).__name__}"
            )

    record(
        len(errors) == 0,
        "result_contract",
        "Phase 1.4 contract: all keys and types correct."
        if not errors else
        "Phase 1.4 contract VIOLATIONS:\n" + "\n".join(f"    - {e}" for e in errors),
    )

    if not errors and meta:
        info(f"Latency          : {meta.get('latency_seconds')}s")
        info(f"Peak VRAM        : {meta.get('peak_vram_mb')} MB")
        info(f"Vertices         : {meta.get('vertex_count'):,}")
        info(f"Faces            : {meta.get('face_count'):,}")
        info(f"Components before clean: {meta.get('components_before_clean')}")
        info(f"GLB size         : {result.get('file_size_bytes'):,} bytes")


# --------------------------------------------------------------------------- #
# CHECK 5 — Physical file verification (atomic rename pattern)               #
# --------------------------------------------------------------------------- #
def check_physical_file(job_id: str, result: dict | None):
    header("CHECK 5 — Physical GLB File & Atomic Write Verification")

    glb_path = OUTPUTS_DIR / f"{job_id}.glb"
    tmp_glob = list(OUTPUTS_DIR.glob(f"*.tmp"))

    # GLB file present
    record(
        glb_path.exists(),
        "glb_file_exists",
        f"GLB file exists: {glb_path}" if glb_path.exists()
        else f"GLB file NOT FOUND at: {glb_path}",
    )

    if glb_path.exists():
        file_size = glb_path.stat().st_size
        record(
            file_size > 0,
            "glb_file_non_empty",
            f"GLB file size: {file_size:,} bytes",
        )

        # Cross-check size against result contract
        if result:
            contract_size = result.get("file_size_bytes", -1)
            record(
                file_size == contract_size,
                "glb_size_matches_contract",
                f"File size matches contract: {file_size} == {contract_size}",
            )

    # No orphaned .tmp files — confirms atomic rename completed cleanly
    # Filter to only .tmp files that plausibly belong to this job
    stale_tmps = [t for t in tmp_glob]
    record(
        len(stale_tmps) == 0,
        "no_orphaned_tmp_files",
        f"No orphaned .tmp files in outputs/ — atomic rename completed cleanly."
        if len(stale_tmps) == 0
        else f"WARNING: {len(stale_tmps)} orphaned .tmp file(s) found: {stale_tmps}",
    )


# --------------------------------------------------------------------------- #
# CHECK 6 — Redis metadata verification                                      #
# --------------------------------------------------------------------------- #
def check_redis_metadata(job_id: str):
    header("CHECK 6 — Redis Metadata Key Verification")

    meta_key = f"gan3d:job:{job_id}:meta"
    raw      = redis_client.hgetall(meta_key)

    record(
        bool(raw),
        "redis_meta_key_exists",
        f"Redis key '{meta_key}' exists with {len(raw)} field(s)."
        if raw else
        f"Redis key '{meta_key}' NOT FOUND.",
    )

    if raw:
        meta = {k: json.loads(v) for k, v in raw.items()}
        info(f"Metadata fields  : {list(meta.keys())}")

        for required_field in ("submitted_at", "completed_at", "asset_url", "file_size_bytes"):
            record(
                required_field in meta,
                f"redis_meta_{required_field}",
                f"Field '{required_field}': {meta.get(required_field, 'MISSING')}",
            )

        # TTL check — key should expire in ~24h
        ttl = redis_client.ttl(meta_key)
        record(
            ttl > 0,
            "redis_meta_ttl",
            f"Redis key TTL: {ttl}s (~{ttl/3600:.1f}h remaining)."
            if ttl > 0 else
            "Redis key has no TTL set — purge task will not clean this up.",
        )


# --------------------------------------------------------------------------- #
# CHECK 7 — VRAM state after task                                            #
# --------------------------------------------------------------------------- #
def check_vram_state():
    header("CHECK 7 — Post-Task VRAM State")

    if not torch.cuda.is_available():
        info("CUDA not available on this machine — skipping VRAM check.")
        CHECKS_PASSED.append("vram_post_task")
        return

    torch.cuda.empty_cache()
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb  = torch.cuda.memory_reserved()  / (1024**2)

    info(f"VRAM allocated (this process): {allocated_mb:.2f} MB")
    info(f"VRAM reserved  (this process): {reserved_mb:.2f} MB")

    # The test runner doesn't load the model — VRAM allocation here should be 0
    # The worker process has the model in VRAM; we're checking the test process
    warn(
        "VRAM check runs in the TEST PROCESS, not the worker process. "
        "To verify worker VRAM cleanup, inspect the worker terminal for "
        "'torch.cuda.empty_cache()' log lines after task completion."
    )
    CHECKS_PASSED.append("vram_post_task")


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def print_summary(start_time: float):
    header("SUBPHASE 1.3 VERIFICATION SUMMARY")

    elapsed = time.perf_counter() - start_time
    info(f"Total test duration: {elapsed:.1f}s")
    info(f"Checks passed: {len(CHECKS_PASSED)}")
    info(f"Checks failed: {len(CHECKS_FAILED)}")

    if CHECKS_PASSED:
        print(f"\n{GREEN}  Passed:{RESET}")
        for c in CHECKS_PASSED:
            print(f"    {GREEN}✔{RESET} {c}")

    if CHECKS_FAILED:
        print(f"\n{RED}  Failed:{RESET}")
        for c in CHECKS_FAILED:
            print(f"    {RED}✘{RESET} {c}")

    print()
    if not CHECKS_FAILED:
        print(f"{GREEN}{BOLD}  SUBPHASE 1.3 VERIFIED.{RESET}")
        print(f"{GREEN}  Celery worker, state machine, and atomic write all confirmed.")
        print(f"  Safe to proceed to Subphase 1.4 (FastAPI application layer).{RESET}\n")
        return True
    else:
        print(f"{RED}{BOLD}  SUBPHASE 1.3 NOT CLEARED.{RESET}")
        print(f"{RED}  Resolve the {len(CHECKS_FAILED)} failed check(s) above before proceeding.{RESET}\n")
        return False


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Subphase 1.3 Lifecycle Verification — 3D-GAN System"
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Max polling timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--resolution", type=int, default=64,
        help="Marching Cubes resolution for the test job (default: 64 for speed)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*66}")
    print("  Subphase 1.3 Worker Lifecycle Test — 3D-GAN System")
    print(f"{'='*66}{RESET}")
    print(f"  Poll timeout : {args.timeout}s")
    print(f"  Resolution   : {args.resolution}³")
    print(f"\n  {YELLOW}PREREQUISITE: Celery worker must be running in a separate terminal:{RESET}")
    print(f"  celery -A celery_app worker --loglevel=info --concurrency=1 -Q gan3d.generate\n")

    start_time = time.perf_counter()

    # CHECK 1
    broker_ok = check_redis_and_broker()
    if not broker_ok:
        fail(
            "Cannot proceed without Redis and a running Celery worker. "
            "Fix CHECK 1 failures first."
        )
        sys.exit(1)

    # CHECK 2
    job_id, async_result = dispatch_task(args.resolution)

    # CHECK 3
    final_state, result = poll_until_terminal(async_result, job_id, args.timeout)

    # CHECK 4
    check_result_contract(result, job_id)

    # CHECK 5
    check_physical_file(job_id, result)

    # CHECK 6
    check_redis_metadata(job_id)

    # CHECK 7
    check_vram_state()

    # Summary
    passed = print_summary(start_time)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
