# Subphase 1.4: FastAPI Application & Endpoint Implementation

## Overview
This is the final interconnected piece of the backend. The FastAPI application ties together:
- **Celery workers** (from Subphase 1.3)
- **Redis state management** (from Subphase 1.2)
- **TripoSR generation** (from Subphase 1.1)

## Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│   FastAPI    │─────▶│   Celery    │
│  (HTTP)     │◀─────│  (main.py)   │      │   Worker    │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │    Redis     │      │   TripoSR   │
                     │ (Job State)  │      │  (GPU Gen)  │
                     └──────────────┘      └─────────────┘
                            │                      │
                            │                      ▼
                            │              ┌─────────────┐
                            └─────────────▶│  outputs/   │
                                           │  (Static)   │
                                           └─────────────┘
```

## File Structure
```
.
├── main.py                 # FastAPI application (NEW)
├── schemas.py              # Pydantic models (NEW)
├── test_api.py            # Verification script (NEW)
├── requirements_api.txt   # FastAPI dependencies (NEW)
├── celery_app.py          # Celery configuration (Subphase 1.3)
├── tasks.py               # Generation task (Subphase 1.3)
├── config.py              # Redis config (Subphase 1.2)
└── outputs/               # Static file directory (auto-created)
```

## Installation

### 1. Install FastAPI Dependencies
```bash
pip install -r requirements_api.txt --break-system-packages
```

### 2. Verify All Services Are Running
Ensure from previous subphases:
- **Redis**: `redis-server` (default port 6379)
- **Celery Worker**: `celery -A celery_app worker --loglevel=info --pool=solo`

## Running the API

### Start the FastAPI Server
```bash
# Development mode with auto-reload
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. POST /v1/generate
Submit a new 3D generation job.

**Queue Depth Check**: Returns `503 Service Unavailable` if ≥10 tasks are active.

**Request:**
```bash
curl -X POST "http://localhost:8000/v1/generate" \
  -F "image=@test_image.png"
```

**Response (202 Accepted):**
```json
{
  "job_id": "abc123...",
  "status": "QUEUED",
  "poll_url": "/v1/jobs/abc123...",
  "message": "Job queued successfully..."
}
```

**Response (503 Service Unavailable):**
```json
{
  "error": "ServiceUnavailable",
  "message": "Queue is at capacity (10/10). Please retry later."
}
```
Headers: `Retry-After: 60`

### 2. GET /v1/jobs/{id}/status
Lightweight status endpoint for high-frequency polling.

**Request:**
```bash
curl "http://localhost:8000/v1/jobs/abc123.../status"
```

**Response:**
```json
{
  "status": "PROCESSING",
  "progress": 45
}
```

**Status Values** (strict matching with Subphase 1.3):
- `QUEUED` - Waiting in queue
- `PROCESSING` - Currently generating
- `COMPLETED` - Generation successful
- `FAILED` - Generation failed
- `EXPIRED` - Job exceeded TTL (1 hour)

### 3. GET /v1/jobs/{id}
Full job details including asset URL when completed.

**Request:**
```bash
curl "http://localhost:8000/v1/jobs/abc123..."
```

**Response (COMPLETED):**
```json
{
  "job_id": "abc123...",
  "status": "COMPLETED",
  "progress": 100,
  "created_at": "2024-02-25T10:30:00",
  "updated_at": "2024-02-25T10:31:45",
  "asset_url": "/outputs/abc123....glb",
  "file_size_bytes": 1048576,
  "generation_time_seconds": 105.3
}
```

**Response (FAILED):**
```json
{
  "job_id": "abc123...",
  "status": "FAILED",
  "progress": 0,
  "created_at": "2024-02-25T10:30:00",
  "updated_at": "2024-02-25T10:30:15",
  "error_message": "CUDA out of memory"
}
```

### 4. DELETE /v1/jobs/{id}
Abort a job and remove from Redis.

**Request:**
```bash
curl -X DELETE "http://localhost:8000/v1/jobs/abc123..."
```

**Response:**
```json
{
  "job_id": "abc123...",
  "status": "aborted",
  "message": "Job abc123... has been aborted."
}
```

## Static File Access

Generated `.glb` files are served via the `/outputs` static mount:

```bash
# HEAD request to verify file exists
curl -I "http://localhost:8000/outputs/abc123....glb"

# Download the file
curl -O "http://localhost:8000/outputs/abc123....glb"
```

## Verification Script

### Usage
```bash
python test_api.py <path_to_test_image>
```

### Example
```bash
python test_api.py test_image.png
```

### Test Flow
1. **Health Check**: Verifies Redis and Celery connectivity
2. **Submit Job**: POST /v1/generate with test image
3. **Poll Status**: GET /v1/jobs/{id}/status until COMPLETED
4. **Verify Asset**: HEAD request on asset_url, download and validate GLB

### Expected Output
```
============================================================
3D Asset Generation API - Verification Test
============================================================

============================================================
STEP 1: Health Check
============================================================
✓ Health check passed: {'api': 'healthy', 'redis': 'healthy', ...}

============================================================
STEP 2: Submit Generation Job
============================================================
✓ Job submitted successfully!
✓ Job ID: abc123...
✓ Status: QUEUED
✓ Poll URL: /v1/jobs/abc123...

============================================================
STEP 3: Poll Job Status
============================================================
  Attempt 15: Status=PROCESSING, Progress=60%
✓ Job completed! (took 30s)

============================================================
STEP 4: Verify Generated Asset
============================================================
✓ Asset exists! Content-Length: 1048576 bytes
✓ File size matches expected: 1048576 bytes
✓ Valid GLB file detected (glTF magic bytes present)
✓ Asset downloaded successfully to: downloaded_test_asset.glb

============================================================
TEST SUMMARY
============================================================
✓ All tests passed!
ℹ Job ID: abc123...
ℹ Status: COMPLETED
ℹ Asset URL: /outputs/abc123....glb
ℹ File Size: 1048576 bytes
ℹ Generation Time: 105.30s

✨ Subphase 1.4 verification complete! Backend is fully operational.
```

## Interconnection Checklist

### ✅ Core App Integration
- [x] Imports `celery_app` from `celery_app.py`
- [x] Imports `generate_3d_asset` task from `tasks.py`
- [x] Uses Redis client for job state management

### ✅ Static File Mounting
- [x] `outputs/` mounted at `/outputs` using `StaticFiles`
- [x] Generated `.glb` files accessible via URL
- [x] Proper file path construction in responses

### ✅ Queue Depth Check
- [x] Uses `celery_app.control.inspect()` to check active/reserved tasks
- [x] Returns `503` with `Retry-After` header when queue ≥ 10

### ✅ Endpoint Implementation
- [x] `POST /v1/generate` - Dispatches task, returns `202` with `job_id`
- [x] `GET /v1/jobs/{id}/status` - Lightweight status/progress only
- [x] `GET /v1/jobs/{id}` - Full details with `asset_url` and `file_size_bytes`
- [x] `DELETE /v1/jobs/{id}` - Aborts task and removes from Redis

### ✅ Pydantic Schemas
- [x] Strict typing for all request/response models
- [x] Status field uses 5 states: QUEUED, PROCESSING, COMPLETED, FAILED, EXPIRED

### ✅ Verification
- [x] Test script submits image
- [x] Polls until COMPLETED
- [x] HEAD request verifies file existence
- [x] Downloads and validates GLB file

## Troubleshooting

### 503 Service Unavailable
- **Cause**: Queue depth ≥ 10 tasks
- **Solution**: Wait for current jobs to complete, or increase `MAX_QUEUE_DEPTH` in `main.py`

### 404 Job Not Found
- **Cause**: Job expired (TTL = 1 hour) or never existed
- **Solution**: Re-submit the job

### Connection Errors
- **Cause**: Redis or Celery worker not running
- **Solution**: 
  ```bash
  # Check Redis
  redis-cli ping  # Should return PONG
  
  # Check Celery
  celery -A celery_app inspect active
  ```

### CUDA Out of Memory
- **Cause**: GPU memory insufficient (RTX 3050 has 4GB)
- **Solution**: Ensure only one generation runs at a time, or reduce model precision

## Next Steps

✅ **Subphase 1.4 Complete**: Backend is fully interconnected and verified.

➡️ **Next**: Phase 2 - Frontend Development
- React application
- Image upload interface
- Real-time progress tracking
- 3D viewer for generated assets

## Technical Notes

### Queue Depth Implementation
```python
def get_queue_depth() -> int:
    inspector = celery_app.control.inspect()
    active = inspector.active()
    reserved = inspector.reserved()
    return sum(len(tasks) for tasks in (active or {}).values()) + \
           sum(len(tasks) for tasks in (reserved or {}).values())
```

### Job State Flow
```
QUEUED → PROCESSING → COMPLETED
   ↓                      ↓
   └───────────────→ FAILED
                         ↓
                     EXPIRED
```

### Static URL Construction
```python
# In task completion:
output_filename = f"{job_id}.glb"
asset_url = f"/outputs/{output_filename}"

# Client access:
# http://localhost:8000/outputs/{job_id}.glb
```
