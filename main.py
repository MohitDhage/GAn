"""
FastAPI Application - Main entry point for the 3D Asset Generation API.
Interconnects Celery workers, Redis state management, and exposes HTTP endpoints.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import redis
import mimetypes

# Fix for .vox files not being recognized as GLB by some browsers
mimetypes.add_type("model/gltf-binary", ".vox")
mimetypes.add_type("model/gltf-binary", ".glb")

# Import our Celery app and task
from celery_app import celery_app
from celery.result import AsyncResult
from tasks import generate_3d_asset, read_job_meta
from schemas import (
    GenerateResponse,
    JobStatusResponse,
    JobDetailResponse,
    DeleteJobResponse,
    ErrorResponse,
)

# Initialize FastAPI
app = FastAPI(
    title="3D Asset Generation API",
    description="AI-powered 3D asset generation from 2D images using TripoSR",
    version="1.0.0",
)

# CORS middleware for frontend integration (Phase 2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection for job state management
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)

# Static file serving for generated assets
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Constants
MAX_QUEUE_DEPTH = 10
JOB_TTL_SECONDS = 3600  # Jobs expire after 1 hour


def get_queue_depth() -> int:
    """
    Check the number of active and reserved tasks in Celery.
    Returns the total count of tasks currently being processed or queued.
    """
    inspector = celery_app.control.inspect()
    
    # Get active tasks (currently processing)
    active = inspector.active()
    active_count = sum(len(tasks) for tasks in (active or {}).values())
    
    # Get reserved tasks (queued and ready to process)
    reserved = inspector.reserved()
    reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())
    
    return active_count + reserved_count


def get_job_data(job_id: str) -> Optional[dict]:
    """Retrieve job data from Redis"""
    data = redis_client.get(f"job:{job_id}")
    if data:
        return json.loads(data)
    return None


def update_job_data(job_id: str, updates: dict):
    """Update job data in Redis"""
    data = get_job_data(job_id)
    if data:
        data.update(updates)
        data["updated_at"] = datetime.utcnow().isoformat()
        redis_client.setex(
            f"job:{job_id}",
            JOB_TTL_SECONDS,
            json.dumps(data)
        )


def delete_job_data(job_id: str):
    """Delete job data from Redis"""
    redis_client.delete(f"job:{job_id}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "3D Asset Generation API",
        "status": "operational",
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check including Redis and Celery connectivity.
    """
    health = {
        "api": "healthy",
        "redis": "unknown",
        "celery": "unknown",
        "queue_depth": 0,
    }
    
    # Check Redis
    try:
        redis_client.ping()
        health["redis"] = "healthy"
    except Exception as e:
        health["redis"] = f"unhealthy: {str(e)}"
    
    # Check Celery
    try:
        queue_depth = get_queue_depth()
        health["celery"] = "healthy"
        health["queue_depth"] = queue_depth
    except Exception as e:
        health["celery"] = f"unhealthy: {str(e)}"
    
    return health


@app.post(
    "/v1/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Job queued successfully"},
        503: {"description": "Service unavailable - queue full", "model": ErrorResponse},
        400: {"description": "Invalid request", "model": ErrorResponse},
    },
)
async def create_generation_job(
    image: UploadFile = File(..., description="Input image file (PNG, JPG, JPEG)"),
    export_format: str = "glb",
    skin_removal_layers: int = 0
):
    """
    Submit a new 3D asset generation job.
    
    Queue depth check: Returns 503 if >= 10 tasks are active/reserved.
    """
    # Queue depth check
    try:
        current_depth = get_queue_depth()
        if current_depth >= MAX_QUEUE_DEPTH:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "ServiceUnavailable",
                    "message": f"Queue is at capacity ({current_depth}/{MAX_QUEUE_DEPTH}). Please retry later.",
                },
                headers={"Retry-After": "60"},  # Suggest retry in 60 seconds
            )
    except Exception as e:
        # If we can't check queue depth, log but don't block submission
        print(f"Warning: Could not check queue depth: {e}")
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an image (PNG, JPG, JPEG).",
        )
    
    # Read image data
    try:
        import uuid
        import io
        import numpy as np
        from PIL import Image

        image_bytes = await image.read()

        # Open and preprocess image — convert to float32 tensor (1, 3, H, W)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_image, dtype=np.float32) / 255.0  # H x W x 3, [0,1]
        img_np = img_np.transpose(2, 0, 1)                       # 3 x H x W
        img_np = img_np[np.newaxis, ...]                         # 1 x 3 x H x W

        image_shape = list(img_np.shape)          # [1, 3, H, W]
        image_data  = img_np.flatten().tolist()   # flat float list (JSON-safe)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image file: {str(e)}",
        )

    # Generate a stable job_id before dispatch so Redis key matches task ID
    job_id = str(uuid.uuid4())

    # Dispatch Celery task — pass job_id as both first arg AND task_id
    task = generate_3d_asset.apply_async(
        args=[job_id, image_data, image_shape, None, 128, export_format, skin_removal_layers],
        task_id=job_id,
    )
    
    # Store initial job state in Redis
    job_data = {
        "job_id": job_id,
        "status": "QUEUED",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "filename": image.filename,
    }
    redis_client.setex(
        f"job:{job_id}",
        JOB_TTL_SECONDS,
        json.dumps(job_data)
    )
    
    # Build poll URL
    poll_url = f"/v1/jobs/{job_id}"
    
    return GenerateResponse(
        job_id=job_id,
        status="QUEUED",
        poll_url=poll_url,
        message="Job queued successfully. Use poll_url to check status.",
    )


@app.get(
    "/v1/jobs/{job_id}/status",
    response_model=JobStatusResponse,
    responses={
        200: {"description": "Job status retrieved"},
        404: {"description": "Job not found", "model": ErrorResponse},
    },
)
async def get_job_status(job_id: str):
    """
    Lightweight endpoint for high-frequency polling.
    Returns only status and progress.
    """
    # 1. Check Redis for base data (existence check)
    job_data = get_job_data(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or expired.",
        )
    
    # 2. Query Celery for live status
    result = AsyncResult(job_id, app=celery_app)
    
    # Map Celery states to our JobStatus enum
    # STARTED/PENDING -> QUEUED or PROCESSING
    # SUCCESS -> COMPLETED
    # FAILURE -> FAILED
    status_map = {
        "PENDING": "QUEUED",
        "STARTED": "PROCESSING",
        "PROGRESS": "PROCESSING",
        "SUCCESS": "COMPLETED",
        "FAILURE": "FAILED",
        "REVOKED": "FAILED",
    }
    
    live_status = status_map.get(result.state, "PROCESSING")
    progress = 0
    
    # Extract progress from Celery info if available
    if result.state == "PROGRESS":
        progress = result.info.get("progress", 0)
    elif result.state == "SUCCESS":
        progress = 100
    elif result.state == "STARTED":
        progress = 5
        
    return JobStatusResponse(
        status=live_status,
        progress=progress,
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobDetailResponse,
    responses={
        200: {"description": "Job details retrieved"},
        404: {"description": "Job not found", "model": ErrorResponse},
    },
)
async def get_job_details(job_id: str):
    """
    Get full job details including asset URL and file size when completed.
    """
    # 1. Base data from Redis (DB 0)
    job_data = get_job_data(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or expired.",
        )
    
    # 2. Live status from Celery
    result = AsyncResult(job_id, app=celery_app)
    
    # 3. Enriched metadata from tasks (DB 1)
    meta = read_job_meta(job_id)
    
    # Map status
    status_map = {
        "PENDING": "QUEUED",
        "STARTED": "PROCESSING",
        "PROGRESS": "PROCESSING",
        "PROCESSING": "PROCESSING",
        "SUCCESS": "COMPLETED",
        "FAILURE": "FAILED",
        "REVOKED": "FAILED",
    }
    
    current_status = status_map.get(result.state, job_data["status"])
    progress = 0
    if result.state in ["PROGRESS", "PROCESSING"]:
        if isinstance(result.info, dict):
            progress = result.info.get("progress", 0)
    elif result.state == "SUCCESS":
        progress = 100
        
    # Build response
    response = JobDetailResponse(
        job_id=job_id,
        status=current_status,
        progress=progress,
        created_at=datetime.fromisoformat(job_data["created_at"]),
        updated_at=datetime.fromisoformat(job_data["updated_at"]),
    )
    
    # Add details from Celery result if successful
    if result.state == "SUCCESS" and result.result is not None:
        response.asset_url = result.result.get("asset_url")
        response.voxel_grid_url = result.result.get("voxel_grid_url")
        response.radiography_url = result.result.get("radiography_url") # Added radiography_url
        response.file_size_bytes = result.result.get("file_size_bytes")
        # Assuming latency_seconds might be nested under 'metadata' in result.result
        response.generation_time_seconds = result.result.get("metadata", {}).get("latency_seconds")
    
    # Add details from meta (DB 1) - this might contain error messages or other info
    # that isn't directly in the task result, or for non-SUCCESS states.
    if meta:
        if "asset_url" in meta and not response.asset_url: # Only set if not already from result
            response.asset_url = meta["asset_url"]
        if "voxel_grid_url" in meta and not response.voxel_grid_url:
            response.voxel_grid_url = meta["voxel_grid_url"]
        if "radiography_url" in meta and not response.radiography_url: # Added radiography_url
            response.radiography_url = meta["radiography_url"]
        if "file_size_bytes" in meta and not response.file_size_bytes:
            response.file_size_bytes = meta["file_size_bytes"]
        if "latency_seconds" in meta and not response.generation_time_seconds:
            response.generation_time_seconds = meta["latency_seconds"]
        if "error" in meta:
            response.error_message = meta["error"].get("message")
    
    return response


@app.delete(
    "/v1/jobs/{job_id}",
    response_model=DeleteJobResponse,
    responses={
        200: {"description": "Job deleted/aborted"},
        404: {"description": "Job not found", "model": ErrorResponse},
    },
)
async def delete_job(job_id: str):
    """
    Abort a job and remove it from Redis.
    Note: If the task is already processing, it may not be immediately aborted.
    """
    job_data = get_job_data(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or expired.",
        )
    
    # Try to revoke the Celery task
    try:
        celery_app.control.revoke(job_id, terminate=True, signal="SIGKILL")
        revoke_status = "aborted"
    except Exception as e:
        print(f"Warning: Could not revoke task {job_id}: {e}")
        revoke_status = "deletion_requested"
    
    # Delete from Redis
    delete_job_data(job_id)
    
    # Clean up generated file if it exists
    if job_data.get("output_path"):
        output_path = Path(job_data["output_path"])
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete file {output_path}: {e}")
    
    return DeleteJobResponse(
        job_id=job_id,
        status=revoke_status,
        message=f"Job {job_id} has been {revoke_status}.",
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )