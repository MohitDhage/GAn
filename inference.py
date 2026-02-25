"""
==============================================================================
Subphase 1.2 — GAN Inference Pipeline
3D-GAN Generation System

File        : inference.py
Hardware    : NVIDIA RTX 3050 (4GB VRAM Laptop GPU)
CUDA        : 11.8  |  PyTorch : 2.1.2
Python      : 3.10+

Public API (consumed by Celery worker in Subphase 1.3):
    run_inference(job_id: str, input_tensor: torch.Tensor, **kwargs) -> dict

Return contract (matches Phase 1.4 /v1/jobs/{id} schema):
    {
        "asset_url"       : str,   # e.g. "/outputs/<job_id>.glb"
        "file_size_bytes" : int,
        "metadata"        : {
            "job_id"           : str,
            "resolution"       : int,
            "latency_seconds"  : float,
            "peak_vram_mb"     : float,
            "vertex_count"     : int,
            "face_count"       : int,
            "components_before_clean": int,
            "generator_class"  : str,
        }
    }

Interconnection notes
---------------------
- OUTPUTS_DIR must match the FastAPI StaticFiles mount path set in Subphase 1.4.
  Default: project_root/outputs  (override via env var OUTPUTS_DIR)
- Job IDs are UUIDs issued by FastAPI POST /v1/generate and passed through
  Celery to this function — never generated here.
- This module is STANDALONE: all functions are independently testable
  without a running FastAPI or Celery process.
==============================================================================
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import trimesh
from skimage.measure import marching_cubes

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

# Resolved at import time so the Celery worker and FastAPI process share the
# same path without additional config wiring.
OUTPUTS_DIR: Path = Path(os.environ.get("OUTPUTS_DIR", Path(__file__).parent / "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Default scalar-field resolution for Marching Cubes.
# 128³ = ~8MB float32 tensor — safe on 4GB VRAM.
# Reduce to 96 or 64 if you see OOM during post-processing.
DEFAULT_MC_RESOLUTION: int = 128

# Marching Cubes iso-surface threshold.
DEFAULT_ISO_LEVEL: float = 0.5

# Latent / noise vector dimensionality fed to the generator.
LATENT_DIM: int = 512

# Image encoder output channels fed into the generator.
IMAGE_FEATURE_DIM: int = 256

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# --------------------------------------------------------------------------- #
# VRAM guard — context manager + decorator                                    #
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def vram_guard(label: str = "operation"):
    """
    Context manager that:
      1. Resets the VRAM peak counter before the block.
      2. Calls torch.cuda.empty_cache() before AND after the block.
      3. Logs peak allocation on exit.

    Usage (context manager):
        with vram_guard("generator_forward_pass"):
            output = model(input)

    Usage (inline decorator on any callable):
        @vram_guard_decorator("my_fn")
        def my_fn(): ...
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.debug("[vram_guard] cache cleared before '%s'", label)

    try:
        yield
    finally:
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb    = peak_bytes / (1024 ** 2)
            torch.cuda.empty_cache()
            logger.info(
                "[vram_guard] '%s' complete. Peak VRAM: %.2f MB. Cache cleared.",
                label, peak_mb,
            )


def vram_guard_decorator(label: str = ""):
    """Decorator form of vram_guard for standalone functions."""
    import functools
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tag = label or fn.__name__
            with vram_guard(tag):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# --------------------------------------------------------------------------- #
# Placeholder GAN Architecture                                                #
# (Replace Generator and ImageEncoder bodies with your trained architecture)  #
# --------------------------------------------------------------------------- #

class ImageEncoder(nn.Module):
    """
    Lightweight CNN that maps a (B, 3, H, W) 2D image to a
    (B, IMAGE_FEATURE_DIM) feature vector.

    Swap this body for your trained encoder when available.
    The output dimensionality (IMAGE_FEATURE_DIM) must stay constant
    so the Generator contract is preserved.
    """
    def __init__(self, out_dim: int = IMAGE_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # → 32 × 64 × 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # → 64 × 32 × 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # → 128 × 16 × 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# → 256 × 8 × 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),                                 # → 256 × 1 × 1
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,3,H,W)
        return self.net(x)                                # → (B, IMAGE_FEATURE_DIM)


class Generator(nn.Module):
    """
    3D volumetric generator.
    Input  : concatenated (image_features + noise_vector) → (B, LATENT_DIM + IMAGE_FEATURE_DIM)
    Output : scalar field of shape (B, 1, R, R, R) where R = mc_resolution.
             Values are sigmoid-normalised to [0, 1] so Marching Cubes
             can use a stable iso_level threshold of 0.5.

    IMPORTANT: this placeholder uses a small MLP for shape only.
    Replace the body with your trained volumetric GAN when available.
    The input/output contract must be preserved.
    """
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        img_feat_dim: int = IMAGE_FEATURE_DIM,
        base_resolution: int = 32,   # internal spatial upscale start
    ):
        super().__init__()
        in_dim = latent_dim + img_feat_dim
        # Project to a small 3D spatial volume then upsample
        self.project = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            # (B, 256, 4, 4, 4) → (B, 128, 8, 8, 8)
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # → (B, 64, 16, 16, 16)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # → (B, 32, 32, 32, 32)
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # → (B, 1, 64, 64, 64)  — final sigmoid for [0,1] field
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        # NOTE: this placeholder outputs a 64³ field regardless of
        # mc_resolution. The scalar_field_to_mesh() function handles
        # interpolation to the requested resolution via numpy.
        self._native_resolution = 64

    def forward(
        self,
        image_features: torch.Tensor,  # (B, IMAGE_FEATURE_DIM)
        noise_vector:   torch.Tensor,  # (B, LATENT_DIM)
    ) -> torch.Tensor:                 # → (B, 1, 64, 64, 64)
        z = torch.cat([image_features, noise_vector], dim=1)
        h = self.project(z)
        h = h.view(-1, 256, 4, 4, 4)
        return self.upsample(h)


# --------------------------------------------------------------------------- #
# Model loader (singleton per worker process)                                 #
# --------------------------------------------------------------------------- #

_encoder_instance:  Optional[ImageEncoder]  = None
_generator_instance: Optional[Generator]    = None


def load_models(device: torch.device) -> tuple[ImageEncoder, Generator]:
    """
    Load (or return cached) encoder and generator onto `device`.
    On the first call the models are instantiated and moved to GPU.
    Subsequent calls within the same worker process return the cached
    instances — critical for worker_concurrency=1 to avoid repeated
    GPU allocation per task.

    To load trained weights, set env vars:
        ENCODER_WEIGHTS_PATH=/path/to/encoder.pth
        GENERATOR_WEIGHTS_PATH=/path/to/generator.pth
    """
    global _encoder_instance, _generator_instance

    if _encoder_instance is not None and _generator_instance is not None:
        return _encoder_instance, _generator_instance

    logger.info("Loading ImageEncoder and Generator onto %s ...", device)

    encoder  = ImageEncoder(out_dim=IMAGE_FEATURE_DIM)
    generator = Generator(latent_dim=LATENT_DIM, img_feat_dim=IMAGE_FEATURE_DIM)

    # Load trained weights if paths are provided
    enc_weights  = os.environ.get("ENCODER_WEIGHTS_PATH")
    gen_weights  = os.environ.get("GENERATOR_WEIGHTS_PATH")

    if enc_weights and Path(enc_weights).exists():
        state = torch.load(enc_weights, map_location=device)
        encoder.load_state_dict(state)
        logger.info("Loaded encoder weights from %s", enc_weights)
    else:
        logger.warning(
            "ENCODER_WEIGHTS_PATH not set or file missing. "
            "Running with randomly initialised encoder (placeholder mode)."
        )

    if gen_weights and Path(gen_weights).exists():
        state = torch.load(gen_weights, map_location=device)
        generator.load_state_dict(state)
        logger.info("Loaded generator weights from %s", gen_weights)
    else:
        logger.warning(
            "GENERATOR_WEIGHTS_PATH not set or file missing. "
            "Running with randomly initialised generator (placeholder mode)."
        )

    encoder.to(device).eval()
    generator.to(device).eval()

    _encoder_instance  = encoder
    _generator_instance = generator

    logger.info("Models loaded and set to eval mode on %s.", device)
    return _encoder_instance, _generator_instance


# --------------------------------------------------------------------------- #
# Scalar field → mesh (Marching Cubes + cleanup)                              #
# --------------------------------------------------------------------------- #

def scalar_field_to_mesh(
    scalar_field: np.ndarray,
    mc_resolution: int = DEFAULT_MC_RESOLUTION,
    iso_level: float   = DEFAULT_ISO_LEVEL,
) -> trimesh.Trimesh:
    """
    Convert a 3D scalar field (numpy array) to a cleaned trimesh.Trimesh.

    Steps
    -----
    1.  Resize / interpolate the field to mc_resolution³ if needed.
    2.  Run skimage.measure.marching_cubes at iso_level.
    3.  Build a trimesh.Trimesh.
    4.  Remove degenerate faces (zero-area triangles).
    5.  Keep only the largest connected component (removes floating noise).

    Parameters
    ----------
    scalar_field  : shape (D, H, W), values in [0, 1]
    mc_resolution : target grid size on each axis before MC
    iso_level     : iso-surface threshold

    Returns
    -------
    trimesh.Trimesh — cleaned, single-component mesh
    """
    # --- 1. Resize to target resolution if needed ---------------------------
    if scalar_field.shape != (mc_resolution, mc_resolution, mc_resolution):
        from scipy.ndimage import zoom
        current_res = scalar_field.shape[0]
        scale = mc_resolution / current_res
        logger.debug(
            "Resampling scalar field from %d³ to %d³ (zoom=%.3f)",
            current_res, mc_resolution, scale,
        )
        scalar_field = zoom(scalar_field, zoom=scale, order=1)  # bilinear

    # Guard: ensure the iso_level is reachable in the field
    field_min, field_max = scalar_field.min(), scalar_field.max()
    if not (field_min < iso_level < field_max):
        # Clamp iso_level to midpoint of the actual range as fallback
        iso_level = float((field_min + field_max) / 2.0)
        logger.warning(
            "iso_level %.3f outside field range [%.3f, %.3f]. "
            "Clamped to midpoint: %.3f",
            DEFAULT_ISO_LEVEL, field_min, field_max, iso_level,
        )

    # --- 2. Marching Cubes --------------------------------------------------
    logger.info(
        "Running Marching Cubes | resolution=%d³ | iso_level=%.3f",
        mc_resolution, iso_level,
    )
    verts, faces, normals, _ = marching_cubes(
        scalar_field,
        level=iso_level,
        allow_degenerate=False,
    )

    logger.info(
        "Marching Cubes output: %d vertices, %d faces",
        len(verts), len(faces),
    )

    # --- 3. Build trimesh ---------------------------------------------------
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # --- 4. Remove degenerate faces ----------------------------------------
    #   trimesh marks degenerate faces (zero area) internally; remove them.
    mask = mesh.nondegenerate_faces()
    n_degenerate = (~mask).sum()
    if n_degenerate > 0:
        mesh.update_faces(mask)
        logger.info("Removed %d degenerate faces.", n_degenerate)

    # --- 5. Largest connected component filter -----------------------------
    components = mesh.split(only_watertight=False)
    n_components = len(components)
    logger.info("Connected components before cleanup: %d", n_components)

    if n_components == 0:
        raise ValueError(
            "Marching Cubes produced no connected components. "
            "The scalar field may not contain a valid iso-surface at the "
            f"given iso_level ({iso_level:.3f}). "
            "Try reducing mc_resolution or adjusting the GAN output range."
        )

    if n_components > 1:
        # Keep the component with the most faces (highest geometric detail)
        mesh = max(components, key=lambda m: len(m.faces))
        logger.info(
            "Kept largest component: %d vertices, %d faces (discarded %d fragment(s)).",
            len(mesh.vertices), len(mesh.faces), n_components - 1,
        )

    return mesh, n_components  # return n_components for metadata


# --------------------------------------------------------------------------- #
# Atomic write                                                                 #
# --------------------------------------------------------------------------- #

def atomic_write_glb(mesh: trimesh.Trimesh, job_id: str) -> tuple[Path, int]:
    """
    Export mesh to GLB using a .tmp → os.rename() atomic pattern.

    Guarantees that the final .glb path is only visible to the filesystem
    (and therefore to FastAPI's static file server) after the file handle
    is fully flushed and closed.

    Returns (final_path, file_size_bytes).
    Raises RuntimeError on write failure with cleanup.
    """
    final_path = OUTPUTS_DIR / f"{job_id}.glb"
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp", dir=OUTPUTS_DIR)
    tmp_path = Path(tmp_path_str)

    try:
        # Close the fd before trimesh writes — trimesh uses its own file handle
        os.close(tmp_fd)

        # CRITICAL: pass file_type explicitly — trimesh infers format from the
        # file extension by default. A .tmp extension has no registered exporter
        # and raises ValueError. Forcing 'glb' bypasses extension detection entirely.
        mesh.export(tmp_path_str, file_type="glb")

        # Validate the tmp file is non-empty before rename
        tmp_size = tmp_path.stat().st_size
        if tmp_size == 0:
            raise RuntimeError(
                f"trimesh exported 0-byte GLB for job {job_id}. "
                "Mesh may be empty after cleanup step."
            )

        # No explicit fsync needed: trimesh opens, writes, and closes its own
        # handle before returning. Re-opening for fsync in 'rb' mode raises
        # [Errno 9] Bad file descriptor on Windows because FlushFileBuffers
        # requires write access. The size check above is the integrity gate.

        # Atomic rename — file is either fully present or absent on any reader
        os.rename(tmp_path_str, final_path)

        file_size = final_path.stat().st_size
        logger.info(
            "GLB written atomically: %s (%d bytes)", final_path, file_size
        )
        return final_path, file_size

    except Exception as exc:
        # Clean up orphaned .tmp on any failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"atomic_write_glb failed for job {job_id}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Public API — run_inference                                                   #
# --------------------------------------------------------------------------- #

def run_inference(
    job_id:         str,
    input_tensor:   torch.Tensor,          # (1, 3, H, W) float32, values [0,1]
    noise_vector:   Optional[torch.Tensor] = None,  # (1, LATENT_DIM); random if None
    mc_resolution:  int   = DEFAULT_MC_RESOLUTION,
    iso_level:      float = DEFAULT_ISO_LEVEL,
) -> dict:
    """
    Full 2D-image → .glb pipeline.

    Parameters
    ----------
    job_id        : UUID string from FastAPI POST /v1/generate
    input_tensor  : preprocessed 2D image tensor, shape (1, 3, H, W), range [0,1]
    noise_vector  : optional latent noise (1, LATENT_DIM); sampled from N(0,1) if None
    mc_resolution : Marching Cubes grid resolution per axis (default 128)
    iso_level     : iso-surface threshold for Marching Cubes (default 0.5)

    Returns (Phase 1.4 contract)
    ----------------------------
    {
        "asset_url"       : "/outputs/<job_id>.glb",
        "file_size_bytes" : int,
        "metadata"        : {
            "job_id"                  : str,
            "resolution"              : int,
            "latency_seconds"         : float,
            "peak_vram_mb"            : float,
            "vertex_count"            : int,
            "face_count"              : int,
            "components_before_clean" : int,
            "generator_class"         : str,
        }
    }

    Raises
    ------
    RuntimeError  — on OOM, mesh export failure, or invalid scalar field
    ValueError    — on invalid input tensor shape
    """
    if input_tensor.dim() != 4 or input_tensor.shape[1] != 3:
        raise ValueError(
            f"input_tensor must be shape (1, 3, H, W), got {tuple(input_tensor.shape)}"
        )

    wall_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "run_inference | job=%s | device=%s | mc_resolution=%d",
        job_id, device, mc_resolution,
    )

    # Reset VRAM peak counter for this job
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ------------------------------------------------------------------ #
    # Step 1 — Load models (cached after first call in worker process)    #
    # ------------------------------------------------------------------ #
    encoder, generator = load_models(device)

    # ------------------------------------------------------------------ #
    # Step 2 — Prepare inputs                                             #
    # ------------------------------------------------------------------ #
    img = input_tensor.to(device)

    if noise_vector is None:
        noise_vector = torch.randn(1, LATENT_DIM, device=device)
        logger.debug("Sampled random noise vector (LATENT_DIM=%d).", LATENT_DIM)
    else:
        noise_vector = noise_vector.to(device)

    # ------------------------------------------------------------------ #
    # Step 3 — Forward pass under FP16 AMP + VRAM guard                  #
    # ------------------------------------------------------------------ #
    with vram_guard("encoder_generator_forward_pass"):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                image_features = encoder(img)              # (1, IMAGE_FEATURE_DIM)
                scalar_field_tensor = generator(           # (1, 1, D, H, W)
                    image_features, noise_vector
                )

    # Move to CPU immediately to free GPU memory before Marching Cubes
    scalar_field_np: np.ndarray = (
        scalar_field_tensor[0, 0].float().cpu().numpy()
    )
    del scalar_field_tensor, image_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Record peak VRAM after the forward pass while stats are still valid
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_vram_mb = peak_vram_bytes / (1024 ** 2)
    logger.info("Peak VRAM after forward pass: %.2f MB", peak_vram_mb)

    # ------------------------------------------------------------------ #
    # Step 4 — Marching Cubes + mesh cleanup (CPU, no VRAM impact)       #
    # ------------------------------------------------------------------ #
    mesh, n_components = scalar_field_to_mesh(
        scalar_field_np,
        mc_resolution=mc_resolution,
        iso_level=iso_level,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Atomic GLB write                                           #
    # ------------------------------------------------------------------ #
    final_path, file_size_bytes = atomic_write_glb(mesh, job_id)

    wall_elapsed = time.perf_counter() - wall_start
    logger.info(
        "run_inference complete | job=%s | latency=%.2fs | file_size=%d bytes",
        job_id, wall_elapsed, file_size_bytes,
    )

    # ------------------------------------------------------------------ #
    # Step 6 — Return Phase 1.4 contract                                 #
    # ------------------------------------------------------------------ #
    return {
        "asset_url":        f"/outputs/{job_id}.glb",
        "file_size_bytes":  file_size_bytes,
        "metadata": {
            "job_id":                   job_id,
            "resolution":               mc_resolution,
            "latency_seconds":          round(wall_elapsed, 3),
            "peak_vram_mb":             round(peak_vram_mb, 2),
            "vertex_count":             len(mesh.vertices),
            "face_count":               len(mesh.faces),
            "components_before_clean":  n_components,
            "generator_class":          generator.__class__.__name__,
        },
    }