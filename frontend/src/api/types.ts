export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'EXPIRED';

export interface GenerateResponse {
    job_id: string;
    status: JobStatus;
    poll_url: string;
    message: string;
}

export interface JobStatusResponse {
    status: JobStatus;
    progress: number;
}

export interface JobDetailResponse {
    job_id: string;
    status: JobStatus;
    progress: number;
    created_at: string;
    updated_at: string;
    asset_url?: string;
    voxel_grid_url?: string;
    radiography_url?: string;
    file_size_bytes?: number;
    generation_time_seconds?: number;
    error_message?: string;
}

export interface ErrorResponse {
    error: string;
    message: string;
}
