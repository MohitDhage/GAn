#!/usr/bin/env python3
"""
==============================================================================
Subphase 1.1 — Smoke Test Script
3D-GAN Generation System

Validates:
  1. Python version (>=3.10 required)
  2. PyTorch installation and version
  3. CUDA availability (torch.cuda.is_available())
  4. Correct CUDA version binding
  5. RTX 3050 device detection and VRAM capacity logging
  6. FP16 (AMP) tensor operation on GPU — validates mixed precision readiness
  7. torch.cuda.empty_cache() executes without error
  8. Redis connectivity on localhost:6379
  9. Celery can import and recognise the Redis broker URL
 10. trimesh GLB export sanity check (CPU-only, no GPU needed)

Run INSIDE the activated virtual environment on the target machine:
    python smoke_test.py

All checks emit structured log lines. Final summary exits with code 0
(all pass) or code 1 (any failure) so it can be used in CI.
==============================================================================
"""

import sys
import os
import json
import time
import traceback

# --------------------------------------------------------------------------- #
# Minimal structured logger (structlog not required at this stage)            #
# --------------------------------------------------------------------------- #

def log(level: str, check: str, message: str, **kwargs):
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "check": check,
        "message": message,
        **kwargs,
    }
    print(json.dumps(record), flush=True)


PASS  = "PASS"
FAIL  = "FAIL"
INFO  = "INFO"
WARN  = "WARN"

results: dict[str, bool] = {}

# --------------------------------------------------------------------------- #
# CHECK 1 — Python version                                                    #
# --------------------------------------------------------------------------- #
def check_python():
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major == 3 and minor >= 10:
        log(PASS, "python_version", f"Python {version_str} — OK", version=version_str)
        results["python_version"] = True
    else:
        log(FAIL, "python_version",
            f"Python {version_str} detected. Project requires >=3.10.",
            version=version_str)
        results["python_version"] = False

# --------------------------------------------------------------------------- #
# CHECK 2 & 3 — PyTorch import + CUDA availability                            #
# --------------------------------------------------------------------------- #
def check_torch_and_cuda():
    try:
        import torch
        log(INFO, "torch_import", f"PyTorch {torch.__version__} imported successfully.",
            torch_version=torch.__version__)
        results["torch_import"] = True

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            log(PASS, "cuda_available", "torch.cuda.is_available() → True")
            results["cuda_available"] = True
        else:
            log(FAIL, "cuda_available",
                "torch.cuda.is_available() → False. "
                "Verify CUDA 11.8 drivers and the cu118 PyTorch wheel are installed.")
            results["cuda_available"] = False
            return  # no point running GPU checks without CUDA

        # CHECK 4 — CUDA version binding
        cuda_version = torch.version.cuda
        if cuda_version and cuda_version.startswith("11.8"):
            log(PASS, "cuda_version", f"CUDA runtime version: {cuda_version}",
                cuda_version=cuda_version)
        else:
            log(WARN, "cuda_version",
                f"Expected CUDA 11.8.x, found {cuda_version}. "
                "Mismatch may cause instability on RTX 3050.",
                cuda_version=cuda_version)
        results["cuda_version"] = True  # warn only, not fatal

        # CHECK 5 — Device detection and VRAM capacity
        device_count = torch.cuda.device_count()
        log(INFO, "device_count", f"{device_count} CUDA device(s) detected.",
            device_count=device_count)

        for idx in range(device_count):
            device_name = torch.cuda.get_device_name(idx)
            vram_bytes   = torch.cuda.get_device_properties(idx).total_memory
            vram_gb      = vram_bytes / (1024 ** 3)

            log(INFO, "device_info",
                f"Device {idx}: {device_name} | VRAM: {vram_gb:.2f} GB",
                device_index=idx,
                device_name=device_name,
                vram_bytes=vram_bytes,
                vram_gb=round(vram_gb, 2),
            )

            if vram_gb < 6.0:
                log(WARN, "vram_warning",
                    f"VRAM ({vram_gb:.2f} GB) is below recommended 8GB. "
                    "FP16 mixed precision is MANDATORY. Monitor OOM risk closely.",
                    vram_gb=round(vram_gb, 2))
            elif vram_gb >= 7.5:
                log(PASS, "vram_capacity",
                    f"VRAM capacity ({vram_gb:.2f} GB) meets 8GB RTX 3050 target.",
                    vram_gb=round(vram_gb, 2))

            if "3050" in device_name:
                log(PASS, "target_device", "RTX 3050 confirmed as active compute device.")
            else:
                log(WARN, "target_device",
                    f"Active device is '{device_name}', not RTX 3050. "
                    "VRAM budgets and concurrency limits were tuned for RTX 3050.",
                    device_name=device_name)

        results["device_info"] = True

        # CHECK 6 — FP16 AMP tensor operation
        try:
            device = torch.device("cuda:0")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                a = torch.randn(512, 512, device=device, dtype=torch.float32)
                b = torch.randn(512, 512, device=device, dtype=torch.float32)
                c = torch.matmul(a, b)
            assert c.shape == (512, 512), "Unexpected output shape from AMP matmul"
            log(PASS, "fp16_amp",
                "FP16 AMP matmul (512x512) completed successfully. "
                "Mixed precision inference is ready.")
            results["fp16_amp"] = True
        except Exception as e:
            log(FAIL, "fp16_amp", f"FP16 AMP test failed: {e}")
            results["fp16_amp"] = False

        # CHECK 7 — torch.cuda.empty_cache()
        try:
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated(0)
            mem_reserved  = torch.cuda.memory_reserved(0)
            log(PASS, "empty_cache",
                "torch.cuda.empty_cache() executed without error.",
                memory_allocated_bytes=mem_allocated,
                memory_reserved_bytes=mem_reserved)
            results["empty_cache"] = True
        except Exception as e:
            log(FAIL, "empty_cache", f"torch.cuda.empty_cache() raised: {e}")
            results["empty_cache"] = False

    except ImportError as e:
        log(FAIL, "torch_import",
            f"Failed to import torch: {e}. "
            "Install via: pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118")
        results["torch_import"] = False
        results["cuda_available"] = False

# --------------------------------------------------------------------------- #
# CHECK 8 — Redis connectivity                                                #
# --------------------------------------------------------------------------- #
def check_redis():
    try:
        import redis as redis_lib
        client = redis_lib.Redis(host="localhost", port=6379, socket_connect_timeout=3)
        pong = client.ping()
        if pong:
            server_info = client.info("server")
            redis_version = server_info.get("redis_version", "unknown")
            log(PASS, "redis_connectivity",
                f"Redis PING → PONG. Server version: {redis_version}",
                redis_version=redis_version,
                host="localhost",
                port=6379)
            results["redis_connectivity"] = True
        else:
            log(FAIL, "redis_connectivity", "Redis PING returned falsy. Check Docker container.")
            results["redis_connectivity"] = False
    except Exception as e:
        log(FAIL, "redis_connectivity",
            f"Cannot connect to Redis on localhost:6379 — {e}. "
            "Start with: docker compose up -d")
        results["redis_connectivity"] = False

# --------------------------------------------------------------------------- #
# CHECK 9 — Celery broker URL resolution                                      #
# --------------------------------------------------------------------------- #
def check_celery():
    try:
        from celery import Celery
        broker_url = "redis://localhost:6379/0"
        backend_url = "redis://localhost:6379/1"
        app = Celery("smoke_test", broker=broker_url, backend=backend_url)
        # Verify the app object is correctly configured without connecting
        assert app.conf.broker_url == broker_url
        assert app.conf.result_backend == backend_url
        log(PASS, "celery_config",
            "Celery app initialised with Redis broker and backend.",
            broker_url=broker_url,
            backend_url=backend_url)
        results["celery_config"] = True
    except ImportError as e:
        log(FAIL, "celery_config", f"Celery import failed: {e}")
        results["celery_config"] = False
    except Exception as e:
        log(FAIL, "celery_config", f"Celery configuration error: {e}")
        results["celery_config"] = False

# --------------------------------------------------------------------------- #
# CHECK 10 — trimesh GLB export                                               #
# --------------------------------------------------------------------------- #
def check_trimesh():
    try:
        import trimesh
        import numpy as np
        import tempfile

        # Minimal valid mesh: a single triangle
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces    = np.array([[0, 1, 2]], dtype=np.int32)
        mesh     = trimesh.Trimesh(vertices=vertices, faces=faces)

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp_path = tmp.name

        mesh.export(tmp_path)
        file_size = os.path.getsize(tmp_path)
        os.unlink(tmp_path)

        if file_size > 0:
            log(PASS, "trimesh_glb_export",
                f"trimesh {trimesh.__version__} GLB export succeeded. "
                f"Output size: {file_size} bytes.",
                trimesh_version=trimesh.__version__,
                glb_size_bytes=file_size)
            results["trimesh_glb_export"] = True
        else:
            log(FAIL, "trimesh_glb_export", "GLB file written but is 0 bytes.")
            results["trimesh_glb_export"] = False

    except Exception as e:
        log(FAIL, "trimesh_glb_export", f"trimesh GLB export failed: {e}\n{traceback.format_exc()}")
        results["trimesh_glb_export"] = False

# --------------------------------------------------------------------------- #
# MAIN — run all checks and emit summary                                      #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70, flush=True)
    print("  Subphase 1.1 Smoke Test — 3D-GAN Generation System", flush=True)
    print("=" * 70, flush=True)

    check_python()
    check_torch_and_cuda()
    check_redis()
    check_celery()
    check_trimesh()

    print("=" * 70, flush=True)

    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    log(INFO, "summary",
        f"{len(passed)} checks passed, {len(failed)} checks failed.",
        passed=passed,
        failed=failed)

    if failed:
        log(FAIL, "final_verdict",
            "SUBPHASE 1.1 NOT CLEARED. Resolve all FAILed checks before proceeding to 1.2.",
            failed_checks=failed)
        sys.exit(1)
    else:
        log(PASS, "final_verdict",
            "SUBPHASE 1.1 CLEARED. All systems nominal. Safe to proceed to Subphase 1.2.")
        sys.exit(0)


if __name__ == "__main__":
    main()
