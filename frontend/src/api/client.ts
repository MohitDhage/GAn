import axios, { type AxiosInstance } from 'axios';
import type { GenerateResponse, JobStatusResponse, JobDetailResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIClient {
    private axiosInstance: AxiosInstance;

    constructor() {
        this.axiosInstance = axios.create({
            baseURL: API_BASE_URL,
            timeout: 30000,
        });
    }

    async submitGenerationJob(image: File, format: string = 'glb', skinRemovalLayers: number = 0): Promise<GenerateResponse> {
        const formData = new FormData();
        formData.append('image', image);

        const url = `/v1/generate?export_format=${format}&skin_removal_layers=${skinRemovalLayers}`;
        const response = await this.axiosInstance.post<GenerateResponse>(url, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        return response.data;
    }

    async getJobStatus(jobId: string): Promise<JobStatusResponse> {
        const response = await this.axiosInstance.get<JobStatusResponse>(`/v1/jobs/${jobId}/status`);
        return response.data;
    }

    async getJobDetails(jobId: string): Promise<JobDetailResponse> {
        const response = await this.axiosInstance.get<JobDetailResponse>(`/v1/jobs/${jobId}`);
        return response.data;
    }

    /**
     * Complex polling logic:
     * 1. Poll /status every 3s
     * 2. When COMPLETED, fetch full details with exponential backoff retry
     */
    async pollJobUntilComplete(
        jobId: string,
        onProgress: (status: JobStatusResponse) => void
    ): Promise<JobDetailResponse> {
        return new Promise((resolve, reject) => {
            const intervalId = setInterval(async () => {
                try {
                    const statusResult = await this.getJobStatus(jobId);
                    onProgress(statusResult);

                    if (statusResult.status === 'COMPLETED') {
                        clearInterval(intervalId);
                        try {
                            const details = await this.fetchWithRetry(() => this.getJobDetails(jobId));
                            resolve(details);
                        } catch (error) {
                            reject(new Error('Failed to fetch job details after completion.'));
                        }
                    } else if (statusResult.status === 'FAILED' || statusResult.status === 'EXPIRED') {
                        clearInterval(intervalId);
                        const finalDetails = await this.getJobDetails(jobId);
                        resolve(finalDetails);
                    }
                } catch (error) {
                    // If status check fails, we might want to log it or continue polling if it's transient
                    console.error('Polling error:', error);
                }
            }, 3000);
        });
    }

    private async fetchWithRetry<T>(fn: () => Promise<T>, retries = 3, delay = 1000): Promise<T> {
        try {
            return await fn();
        } catch (error) {
            if (retries <= 0) throw error;
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.fetchWithRetry(fn, retries - 1, delay * 2);
        }
    }
}

export const apiClient = new APIClient();
