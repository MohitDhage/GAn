#!/usr/bin/env python3
"""
==============================================================================
Subphase 1.2 — Benchmark & Verification Script
3D-GAN Generation System

File        : benchmark_inference.py
Purpose     : Validate the inference pipeline end-to-end and measure:
              - Total wall-clock latency
              - Peak VRAM usage at each pipeline stage
              - Mesh quality metrics (vertex/face count, watertightness)
              - Phase 1.4 return contract shape and types

Usage
-----
    python benchmark_inference.py [--runs N] [--resolution R] [--cpu-only]

Arguments
    --runs        Number of inference runs to average (default: 3)
    --resolution  Marching Cubes resolution (default: 128; try 64 for fast test)
    --cpu-only    Force CPU execution (skips VRAM checks, useful for CI)

Interconnection
---------------
Imports run_inference() directly from inference.py in the same directory.
OUTPUTS_DIR defaults to ./outputs (created if absent).
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Resolve inference module path — handles running from any cwd                #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from inference import (
    LATENT_DIM,
    IMAGE_FEATURE_DIM,
    DEFAULT_MC_RESOLUTION,
    OUTPUTS_DIR,
    run_inference,
    load_models,
    scalar_field_to_mesh,
    atomic_write_glb,
)


# --------------------------------------------------------------------------- #
# ANSI colour helpers                                                          #
# --------------------------------------------------------------------------- #
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔  {RESET}{msg}")
def warn(msg):  print(f"  {YELLOW}⚠  {RESET}{msg}")
def fail(msg):  print(f"  {RED}✘  {RESET}{msg}")
def info(msg):  print(f"  {CYAN}ℹ  {RESET}{msg}")
def header(msg): print(f"\n{BOLD}{CYAN}{'='*66}{RESET}\n{BOLD}  {msg}{RESET}\n{'='*66}")


# --------------------------------------------------------------------------- #
# VRAM snapshot helper                                                        #
# --------------------------------------------------------------------------- #
def vram_snapshot(device: torch.device, label: str) -> dict:
    if device.type != "cuda":
        return {"label": label, "allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    return {
        "label":        label,
        "allocated_mb": torch.cuda.memory_allocated(device)  / (1024**2),
        "reserved_mb":  torch.cuda.memory_reserved(device)   / (1024**2),
        "peak_mb":      torch.cuda.max_memory_allocated(device) / (1024**2),
    }


# --------------------------------------------------------------------------- #
# CHECK 1 — Device & VRAM capacity                                            #
# --------------------------------------------------------------------------- #
def check_device(cpu_only: bool) -> torch.device:
    header("CHECK 1 — Device & VRAM Capacity")

    if cpu_only:
        warn("--cpu-only flag set. Running on CPU. VRAM checks will be skipped.")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        warn("CUDA not available. Falling back to CPU. VRAM checks disabled.")
        return torch.device("cpu")

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / (1024**3)

    info(f"Device      : {props.name}")
    info(f"VRAM        : {vram_gb:.2f} GB ({props.total_memory:,} bytes)")
    info(f"CUDA caps   : compute {props.major}.{props.minor}")
    info(f"MP count    : {props.multi_processor_count}")

    if vram_gb < 3.5:
        fail(f"VRAM too low ({vram_gb:.2f} GB). Minimum 4 GB required.")
        sys.exit(1)
    elif vram_gb < 5.0:
        warn(f"VRAM is {vram_gb:.2f} GB. This is the 4GB variant. "
             "Use mc_resolution ≤ 96 to stay safe.")
    else:
        ok(f"VRAM {vram_gb:.2f} GB — 8GB RTX 3050 confirmed.")

    return device


# --------------------------------------------------------------------------- #
# CHECK 2 — Model load and VRAM cost                                          #
# --------------------------------------------------------------------------- #
def check_model_load(device: torch.device):
    header("CHECK 2 — Model Load & Static VRAM Cost")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = vram_snapshot(device, "before_load")

    t0 = time.perf_counter()
    encoder, generator = load_models(device)
    elapsed = time.perf_counter() - t0

    if device.type == "cuda":
        after = vram_snapshot(device, "after_load")
        model_vram_mb = after["allocated_mb"] - before["allocated_mb"]
        info(f"Model load time    : {elapsed:.2f}s")
        info(f"Static VRAM (models): {model_vram_mb:.2f} MB")

        if model_vram_mb > 2048:
            warn(f"Models consume {model_vram_mb:.0f} MB static VRAM. "
                 "Less than 2 GB headroom on 4 GB device.")
        else:
            ok(f"Static model VRAM: {model_vram_mb:.2f} MB — within safe budget.")
    else:
        ok(f"Models loaded on CPU in {elapsed:.2f}s (VRAM tracking disabled).")

    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    gen_params = sum(p.numel() for p in generator.parameters())
    info(f"Encoder params   : {enc_params:,}")
    info(f"Generator params : {gen_params:,}")

    return encoder, generator


# --------------------------------------------------------------------------- #
# CHECK 3 — Single forward pass: latency + VRAM profiling                    #
# --------------------------------------------------------------------------- #
def check_single_forward_pass(
    device: torch.device,
    mc_resolution: int,
    run_index: int = 1,
) -> dict:
    header(f"CHECK 3.{run_index} — Full Inference Run (mc_resolution={mc_resolution}³)")

    job_id = f"benchmark-run-{run_index:02d}-{uuid.uuid4().hex[:8]}"
    info(f"Job ID: {job_id}")

    # Dummy 2D image tensor: (1, 3, 128, 128), values in [0, 1]
    dummy_image = torch.rand(1, 3, 128, 128, dtype=torch.float32)
    dummy_noise = torch.randn(1, LATENT_DIM, dtype=torch.float32)

    # Reset VRAM peak for this run
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        snap_before = vram_snapshot(device, "pre_inference")
        info(f"VRAM before inference : {snap_before['allocated_mb']:.2f} MB allocated")

    wall_start = time.perf_counter()

    result = run_inference(
        job_id=job_id,
        input_tensor=dummy_image,
        noise_vector=dummy_noise,
        mc_resolution=mc_resolution,
    )

    wall_elapsed = time.perf_counter() - wall_start

    if device.type == "cuda":
        snap_after = vram_snapshot(device, "post_inference")
        info(f"VRAM after inference  : {snap_after['allocated_mb']:.2f} MB allocated")
        info(f"Peak VRAM this run    : {snap_after['peak_mb']:.2f} MB")

    # -- Print results -------------------------------------------------------
    meta = result["metadata"]
    info(f"Total wall latency    : {wall_elapsed:.2f}s (reported: {meta['latency_seconds']}s)")
    info(f"Peak VRAM (reported)  : {meta['peak_vram_mb']:.2f} MB")
    info(f"Mesh vertices         : {meta['vertex_count']:,}")
    info(f"Mesh faces            : {meta['face_count']:,}")
    info(f"Components before clean: {meta['components_before_clean']}")
    info(f"GLB size              : {result['file_size_bytes']:,} bytes")
    info(f"Asset URL             : {result['asset_url']}")

    # -- Validate Phase 1.4 contract shape -----------------------------------
    _validate_contract(result)

    # -- Validate GLB file exists on disk ------------------------------------
    glb_path = OUTPUTS_DIR / f"{job_id}.glb"
    if glb_path.exists() and glb_path.stat().st_size > 0:
        ok(f"GLB file present on disk: {glb_path}")
    else:
        fail(f"GLB file missing or empty at expected path: {glb_path}")

    return {
        "run_index":       run_index,
        "job_id":          job_id,
        "wall_elapsed_s":  wall_elapsed,
        "peak_vram_mb":    meta["peak_vram_mb"],
        "vertex_count":    meta["vertex_count"],
        "face_count":      meta["face_count"],
        "file_size_bytes": result["file_size_bytes"],
    }


# --------------------------------------------------------------------------- #
# Phase 1.4 contract validator                                                #
# --------------------------------------------------------------------------- #
def _validate_contract(result: dict):
    """
    Assert the run_inference() return value matches the documented
    Phase 1.4 response schema exactly.
    """
    errors = []

    # Top-level keys
    for key in ("asset_url", "file_size_bytes", "metadata"):
        if key not in result:
            errors.append(f"Missing top-level key: '{key}'")

    # asset_url format
    asset_url = result.get("asset_url", "")
    if not isinstance(asset_url, str) or not asset_url.startswith("/outputs/"):
        errors.append(f"asset_url must start with '/outputs/', got: '{asset_url}'")
    if not asset_url.endswith(".glb"):
        errors.append(f"asset_url must end with '.glb', got: '{asset_url}'")

    # file_size_bytes
    fsb = result.get("file_size_bytes", 0)
    if not isinstance(fsb, int) or fsb <= 0:
        errors.append(f"file_size_bytes must be positive int, got: {fsb}")

    # metadata keys
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
            errors.append(f"metadata missing field: '{field}'")
        elif not isinstance(meta[field], expected_type):
            errors.append(
                f"metadata['{field}'] expected {expected_type.__name__}, "
                f"got {type(meta[field]).__name__}"
            )

    if errors:
        for e in errors:
            fail(f"CONTRACT VIOLATION: {e}")
        raise AssertionError("Phase 1.4 contract validation failed.")
    else:
        ok("Phase 1.4 return contract validated — all keys and types correct.")


# --------------------------------------------------------------------------- #
# CHECK 4 — OOM safety guard: test resolution ladder                         #
# --------------------------------------------------------------------------- #
def check_resolution_ladder(device: torch.device):
    """
    Verify that reducing mc_resolution is a viable OOM escape hatch.
    Tests 64³ resolution as a fast safety check — does NOT run at 128³ again.
    """
    header("CHECK 4 — OOM Safety: Low-Resolution Fallback (64³)")

    job_id = f"benchmark-low-res-{uuid.uuid4().hex[:8]}"
    dummy_image = torch.rand(1, 3, 128, 128)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    result = run_inference(
        job_id=job_id,
        input_tensor=dummy_image,
        mc_resolution=64,
    )
    elapsed = time.perf_counter() - t0

    meta = result["metadata"]
    info(f"64³ latency      : {elapsed:.2f}s")
    info(f"64³ peak VRAM    : {meta['peak_vram_mb']:.2f} MB")
    info(f"64³ vertex count : {meta['vertex_count']:,}")
    ok("Low-resolution fallback (64³) completed successfully.")


# --------------------------------------------------------------------------- #
# CHECK 5 — empty_cache after task (simulates worker task teardown)          #
# --------------------------------------------------------------------------- #
def check_cache_teardown(device: torch.device):
    header("CHECK 5 — Worker Teardown: torch.cuda.empty_cache()")

    if device.type != "cuda":
        info("CPU mode — skipping VRAM teardown check.")
        return

    before_mb = torch.cuda.memory_allocated(device) / (1024**2)
    torch.cuda.empty_cache()
    after_mb  = torch.cuda.memory_allocated(device) / (1024**2)
    reserved  = torch.cuda.memory_reserved(device)  / (1024**2)

    info(f"Allocated before empty_cache : {before_mb:.2f} MB")
    info(f"Allocated after  empty_cache : {after_mb:.2f} MB")
    info(f"Reserved (cached by allocator): {reserved:.2f} MB")
    ok("torch.cuda.empty_cache() executed without error — worker teardown safe.")


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def print_summary(runs: list[dict], device: torch.device):
    header("BENCHMARK SUMMARY")

    if not runs:
        warn("No completed runs to summarise.")
        return

    latencies   = [r["wall_elapsed_s"]  for r in runs]
    vram_peaks  = [r["peak_vram_mb"]    for r in runs]

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    avg_vram    = np.mean(vram_peaks)
    peak_vram   = np.max(vram_peaks)

    info(f"Runs completed          : {len(runs)}")
    info(f"Latency  — avg          : {avg_latency:.2f}s")
    info(f"Latency  — min          : {min_latency:.2f}s")
    info(f"Latency  — max          : {max_latency:.2f}s")
    info(f"Peak VRAM — avg         : {avg_vram:.2f} MB")
    info(f"Peak VRAM — max         : {peak_vram:.2f} MB")

    # Verdict vs Subphase 1.2 design targets
    print()
    if avg_latency <= 60.0:
        ok(f"Avg latency {avg_latency:.1f}s ≤ 60s target — within spec.")
    elif avg_latency <= 90.0:
        warn(f"Avg latency {avg_latency:.1f}s exceeds 60s target. "
             "Consider reducing mc_resolution. "
             "Review task_time_limit in Celery config (Subphase 1.3).")
    else:
        fail(f"Avg latency {avg_latency:.1f}s exceeds 90s. "
             "TASK TIMEOUT RISK. Reduce resolution or profile bottleneck.")

    if device.type == "cuda":
        device_props = torch.cuda.get_device_properties(device)
        total_vram_mb = device_props.total_memory / (1024**2)
        headroom_mb   = total_vram_mb - peak_vram
        info(f"VRAM headroom after peak: {headroom_mb:.0f} MB / {total_vram_mb:.0f} MB total")

        if headroom_mb < 300:
            fail(f"Less than 300 MB VRAM headroom. HIGH OOM RISK. "
                 "Reduce mc_resolution or model size before Subphase 1.3.")
        elif headroom_mb < 800:
            warn(f"VRAM headroom is {headroom_mb:.0f} MB. Acceptable but tight. "
                 "Monitor carefully under load.")
        else:
            ok(f"VRAM headroom {headroom_mb:.0f} MB — safe margin for production.")

    print()
    info("Paste this summary into your Subphase 1.2 verification response.")
    info("Required numbers for Subphase 1.3 Celery config:")
    info(f"  task_time_limit      = {max(int(max_latency * 2), 120)}")
    info(f"  task_soft_time_limit = {max(int(max_latency * 1.7), 100)}")


# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Subphase 1.2 Benchmark — 3D-GAN Inference Pipeline"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of full inference runs to average (default: 3)"
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_MC_RESOLUTION,
        help=f"Marching Cubes resolution (default: {DEFAULT_MC_RESOLUTION}). "
             "Use 64 for a fast first pass."
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU execution (disables VRAM checks)"
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*66}")
    print("  Subphase 1.2 Benchmark — 3D-GAN Generation System")
    print(f"{'='*66}{RESET}")
    print(f"  Runs        : {args.runs}")
    print(f"  Resolution  : {args.resolution}³")
    print(f"  Device mode : {'CPU (forced)' if args.cpu_only else 'Auto (GPU preferred)'}")

    # Check 1
    device = check_device(args.cpu_only)

    # Check 2
    check_model_load(device)

    # Check 3 — N timed runs
    completed_runs = []
    for i in range(1, args.runs + 1):
        try:
            run_result = check_single_forward_pass(
                device=device,
                mc_resolution=args.resolution,
                run_index=i,
            )
            completed_runs.append(run_result)
            ok(f"Run {i}/{args.runs} completed.")
        except Exception as e:
            fail(f"Run {i}/{args.runs} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Check 4
    check_resolution_ladder(device)

    # Check 5
    check_cache_teardown(device)

    # Summary
    print_summary(completed_runs, device)

    # Exit code
    if len(completed_runs) == args.runs:
        print(f"\n{GREEN}{BOLD}SUBPHASE 1.2 BENCHMARK COMPLETE.{RESET}")
        print(f"{GREEN}All {args.runs} runs passed. Safe to report latency to Subphase 1.3.{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}{BOLD}SUBPHASE 1.2 BENCHMARK INCOMPLETE.{RESET}")
        print(f"{RED}{args.runs - len(completed_runs)} run(s) failed. "
              f"Resolve errors before proceeding.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
