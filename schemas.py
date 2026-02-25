"""
Pydantic schemas for API request/response models.
Ensures strict typing and validation across all endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

# Strict state definitions matching Subphase 1.3
JobStatus = Literal["QUEUED", "PROCESSING", "COMPLETED", "FAILED", "EXPIRED"]


class GenerateRequest(BaseModel):
    """Request model for job submission (for documentation purposes)"""
    pass  # File upload handled via FastAPI's UploadFile


class GenerateResponse(BaseModel):
    """Response model for POST /v1/generate"""
    job_id: str = Field(..., description="Unique identifier for the generation job")
    status: JobStatus = Field(..., description="Initial job status (QUEUED)")
    poll_url: str = Field(..., description="URL to poll for job status")
    message: str = Field(..., description="Human-readable status message")


class JobStatusResponse(BaseModel):
    """Lightweight response model for GET /v1/jobs/{id}/status"""
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Completion percentage (0-100)")


class JobDetailResponse(BaseModel):
    """Full response model for GET /v1/jobs/{id}"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Completion percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Populated only on COMPLETED
    asset_url: Optional[str] = Field(None, description="URL to download the generated .glb file")
    file_size_bytes: Optional[int] = Field(None, description="Size of the generated file in bytes")
    generation_time_seconds: Optional[float] = Field(None, description="Time taken to generate")
    
    # Populated only on FAILED
    error_message: Optional[str] = Field(None, description="Error details if status is FAILED")


class DeleteJobResponse(BaseModel):
    """Response model for DELETE /v1/jobs/{id}"""
    job_id: str = Field(..., description="Deleted job identifier")
    status: str = Field(..., description="Deletion status")
    message: str = Field(..., description="Deletion confirmation message")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    job_id: Optional[str] = Field(None, description="Related job ID if applicable")