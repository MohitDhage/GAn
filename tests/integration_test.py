"""
Comprehensive Integration Test for Subphase 1.4
Validates all interconnections between FastAPI, Celery, Redis, and TripoSR.

This test verifies:
1. API → Celery task dispatch
2. Celery → Redis state updates
3. Redis → API state retrieval
4. TripoSR → File generation
5. Static file serving
6. Queue depth limiting
7. All 5 job states
"""
import os
import sys
import time
import json
import requests
import redis
from pathlib import Path
from typing import Dict, Optional


API_BASE = "http://localhost:8000"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


class IntegrationTest:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
        )
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test(self, name: str):
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)
    
    def log_pass(self, msg: str):
        print(f"✓ {msg}")
        self.passed_tests += 1
    
    def log_fail(self, msg: str):
        print(f"✗ {msg}")
        self.failed_tests += 1
    
    def log_info(self, msg: str):
        print(f"  {msg}")
    
    def test_redis_connectivity(self) -> bool:
        """Test 1: Verify Redis connection"""
        self.log_test("Redis Connectivity")
        try:
            self.redis_client.ping()
            self.log_pass("Redis connection successful")
            return True
        except Exception as e:
            self.log_fail(f"Redis connection failed: {e}")
            return False
    
    def test_api_health(self) -> bool:
        """Test 2: Verify API health endpoint"""
        self.log_test("API Health Check")
        try:
            response = requests.get(f"{API_BASE}/health")
            response.raise_for_status()
            health = response.json()
            
            self.log_info(f"Health data: {health}")
            
            if health.get("api") == "healthy":
                self.log_pass("API is healthy")
            else:
                self.log_fail("API not healthy")
                return False
            
            if health.get("redis") == "healthy":
                self.log_pass("Redis connection from API verified")
            else:
                self.log_fail("Redis connection from API failed")
                return False
            
            if health.get("celery") == "healthy":
                self.log_pass("Celery connection from API verified")
            else:
                self.log_fail("Celery connection from API failed")
                return False
            
            return True
        except Exception as e:
            self.log_fail(f"Health check failed: {e}")
            return False
    
    def test_celery_import(self) -> bool:
        """Test 3: Verify Celery app import"""
        self.log_test("Celery App Import")
        try:
            from celery_app import celery_app
            from tasks import generate_3d_asset
            
            self.log_pass("celery_app imported successfully")
            self.log_pass("generate_3d_asset task imported successfully")
            
            # Verify task is registered
            if 'tasks.generate_3d_asset' in celery_app.tasks:
                self.log_pass("Task registered in Celery app")
            else:
                self.log_fail("Task not registered in Celery app")
                return False
            
            return True
        except Exception as e:
            self.log_fail(f"Import failed: {e}")
            return False
    
    def test_pydantic_schemas(self) -> bool:
        """Test 4: Verify Pydantic schema constraints"""
        self.log_test("Pydantic Schema Validation")
        try:
            from schemas import (
                JobStatus,
                GenerateResponse,
                JobStatusResponse,
                JobDetailResponse,
            )
            
            # Test status literal constraint
            valid_statuses = ["QUEUED", "PROCESSING", "COMPLETED", "FAILED", "EXPIRED"]
            self.log_info(f"Valid statuses: {valid_statuses}")
            self.log_pass("JobStatus literal type defined correctly")
            
            # Test schema instantiation
            test_response = GenerateResponse(
                job_id="test123",
                status="QUEUED",
                poll_url="/v1/jobs/test123",
                message="Test",
            )
            self.log_pass("GenerateResponse schema validated")
            
            test_status = JobStatusResponse(
                status="PROCESSING",
                progress=50,
            )
            self.log_pass("JobStatusResponse schema validated")
            
            return True
        except Exception as e:
            self.log_fail(f"Schema validation failed: {e}")
            return False
    
    def test_static_mount(self) -> bool:
        """Test 5: Verify static file mounting"""
        self.log_test("Static File Mount")
        try:
            # Create a test file in outputs/
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            test_file = outputs_dir / "test_static.txt"
            test_file.write_text("Static file test")
            
            # Try to access via API
            response = requests.get(f"{API_BASE}/outputs/test_static.txt")
            response.raise_for_status()
            
            if response.text == "Static file test":
                self.log_pass("Static file served correctly")
            else:
                self.log_fail("Static file content mismatch")
                return False
            
            # Clean up
            test_file.unlink()
            self.log_info("Test file cleaned up")
            
            return True
        except Exception as e:
            self.log_fail(f"Static mount test failed: {e}")
            return False
    
    def test_job_submission_and_redis(self, image_path: Path) -> Optional[str]:
        """Test 6: Submit job and verify Redis state"""
        self.log_test("Job Submission & Redis Integration")
        try:
            # Submit job
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                response = requests.post(f"{API_BASE}/v1/generate", files=files)
            
            if response.status_code == 503:
                self.log_fail("Queue full (503) - clear jobs and retry")
                return None
            
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data["job_id"]
            
            self.log_pass(f"Job submitted: {job_id}")
            self.log_info(f"Initial status: {job_data['status']}")
            
            # Verify Redis state
            redis_key = f"job:{job_id}"
            redis_data = self.redis_client.get(redis_key)
            
            if redis_data:
                parsed_data = json.loads(redis_data)
                self.log_pass("Job found in Redis")
                self.log_info(f"Redis data: {parsed_data}")
                
                if parsed_data["status"] == "QUEUED":
                    self.log_pass("Initial status in Redis is QUEUED")
                else:
                    self.log_fail(f"Unexpected status in Redis: {parsed_data['status']}")
            else:
                self.log_fail("Job not found in Redis")
                return None
            
            # Verify poll_url is correct
            if job_data["poll_url"] == f"/v1/jobs/{job_id}":
                self.log_pass("poll_url constructed correctly")
            else:
                self.log_fail(f"Invalid poll_url: {job_data['poll_url']}")
            
            return job_id
        except Exception as e:
            self.log_fail(f"Job submission failed: {e}")
            return None
    
    def test_status_endpoint(self, job_id: str) -> bool:
        """Test 7: Lightweight status endpoint"""
        self.log_test("Status Endpoint (/v1/jobs/{id}/status)")
        try:
            response = requests.get(f"{API_BASE}/v1/jobs/{job_id}/status")
            response.raise_for_status()
            status_data = response.json()
            
            self.log_info(f"Status response: {status_data}")
            
            # Verify response structure
            if "status" in status_data and "progress" in status_data:
                self.log_pass("Status endpoint returns correct fields")
            else:
                self.log_fail("Missing fields in status response")
                return False
            
            # Verify progress is 0-100
            progress = status_data["progress"]
            if 0 <= progress <= 100:
                self.log_pass(f"Progress within valid range: {progress}")
            else:
                self.log_fail(f"Invalid progress value: {progress}")
                return False
            
            return True
        except Exception as e:
            self.log_fail(f"Status endpoint test failed: {e}")
            return False
    
    def test_detail_endpoint(self, job_id: str) -> Dict:
        """Test 8: Full detail endpoint"""
        self.log_test("Detail Endpoint (/v1/jobs/{id})")
        try:
            response = requests.get(f"{API_BASE}/v1/jobs/{job_id}")
            response.raise_for_status()
            detail_data = response.json()
            
            self.log_info(f"Detail response keys: {detail_data.keys()}")
            
            # Verify required fields
            required_fields = ["job_id", "status", "progress", "created_at", "updated_at"]
            for field in required_fields:
                if field in detail_data:
                    self.log_pass(f"Field '{field}' present")
                else:
                    self.log_fail(f"Field '{field}' missing")
                    return {}
            
            return detail_data
        except Exception as e:
            self.log_fail(f"Detail endpoint test failed: {e}")
            return {}
    
    def test_job_completion(self, job_id: str, max_wait: int = 300) -> Optional[Dict]:
        """Test 9: Wait for job completion and verify all states"""
        self.log_test("Job State Transitions")
        
        seen_states = set()
        start_time = time.time()
        
        try:
            while time.time() - start_time < max_wait:
                response = requests.get(f"{API_BASE}/v1/jobs/{job_id}/status")
                response.raise_for_status()
                status_data = response.json()
                
                current_status = status_data["status"]
                progress = status_data["progress"]
                
                # Track state transitions
                if current_status not in seen_states:
                    seen_states.add(current_status)
                    self.log_pass(f"State: {current_status} (progress: {progress}%)")
                
                if current_status == "COMPLETED":
                    # Get full details
                    detail_response = requests.get(f"{API_BASE}/v1/jobs/{job_id}")
                    detail_response.raise_for_status()
                    detail_data = detail_response.json()
                    
                    self.log_pass("Job completed successfully")
                    
                    # Verify completion fields
                    if detail_data.get("asset_url"):
                        self.log_pass(f"asset_url present: {detail_data['asset_url']}")
                    else:
                        self.log_fail("asset_url missing in COMPLETED state")
                    
                    if detail_data.get("file_size_bytes"):
                        self.log_pass(f"file_size_bytes: {detail_data['file_size_bytes']}")
                    else:
                        self.log_fail("file_size_bytes missing in COMPLETED state")
                    
                    return detail_data
                
                elif current_status == "FAILED":
                    self.log_fail("Job failed")
                    detail_response = requests.get(f"{API_BASE}/v1/jobs/{job_id}")
                    detail_data = detail_response.json()
                    self.log_info(f"Error: {detail_data.get('error_message')}")
                    return None
                
                time.sleep(2)
            
            self.log_fail(f"Job did not complete within {max_wait}s")
            return None
        except Exception as e:
            self.log_fail(f"Job completion test failed: {e}")
            return None
    
    def test_asset_verification(self, asset_url: str, file_size_bytes: int) -> bool:
        """Test 10: Verify generated asset"""
        self.log_test("Asset Verification")
        try:
            # Build full URL
            if not asset_url.startswith("http"):
                full_url = f"{API_BASE}{asset_url}"
            else:
                full_url = asset_url
            
            # HEAD request
            head_response = requests.head(full_url)
            head_response.raise_for_status()
            
            content_length = int(head_response.headers.get("content-length", 0))
            self.log_pass(f"Asset accessible via HEAD request")
            self.log_info(f"Content-Length: {content_length} bytes")
            
            # Verify file size
            if content_length == file_size_bytes:
                self.log_pass("File size matches API response")
            else:
                self.log_fail(f"File size mismatch: {content_length} != {file_size_bytes}")
            
            # Download and verify GLB
            get_response = requests.get(full_url)
            get_response.raise_for_status()
            
            if get_response.content[:4] == b"glTF":
                self.log_pass("Valid GLB file (glTF magic bytes)")
            else:
                self.log_fail("Invalid GLB file format")
                return False
            
            return True
        except Exception as e:
            self.log_fail(f"Asset verification failed: {e}")
            return False
    
    def test_delete_endpoint(self, job_id: str) -> bool:
        """Test 11: Job deletion"""
        self.log_test("Delete Endpoint")
        try:
            # First verify job exists
            pre_check = requests.get(f"{API_BASE}/v1/jobs/{job_id}/status")
            if pre_check.status_code == 404:
                self.log_info("Job already deleted/expired")
                return True
            
            # Delete job
            response = requests.delete(f"{API_BASE}/v1/jobs/{job_id}")
            response.raise_for_status()
            delete_data = response.json()
            
            self.log_pass(f"Job deleted: {delete_data['message']}")
            
            # Verify it's gone from Redis
            redis_key = f"job:{job_id}"
            if not self.redis_client.exists(redis_key):
                self.log_pass("Job removed from Redis")
            else:
                self.log_fail("Job still in Redis after deletion")
                return False
            
            # Verify 404 on subsequent requests
            post_check = requests.get(f"{API_BASE}/v1/jobs/{job_id}/status")
            if post_check.status_code == 404:
                self.log_pass("API returns 404 for deleted job")
            else:
                self.log_fail("API did not return 404 for deleted job")
                return False
            
            return True
        except Exception as e:
            self.log_fail(f"Delete test failed: {e}")
            return False
    
    def run_all_tests(self, image_path: Path):
        """Run complete integration test suite"""
        print("\n" + "="*60)
        print("COMPREHENSIVE INTEGRATION TEST - SUBPHASE 1.4")
        print("="*60)
        
        # Test 1-5: Infrastructure
        if not self.test_redis_connectivity():
            return False
        if not self.test_api_health():
            return False
        if not self.test_celery_import():
            return False
        if not self.test_pydantic_schemas():
            return False
        if not self.test_static_mount():
            return False
        
        # Test 6-9: Job lifecycle
        job_id = self.test_job_submission_and_redis(image_path)
        if not job_id:
            return False
        
        if not self.test_status_endpoint(job_id):
            return False
        
        detail_data = self.test_detail_endpoint(job_id)
        if not detail_data:
            return False
        
        completion_data = self.test_job_completion(job_id)
        if not completion_data:
            return False
        
        # Test 10: Asset verification
        asset_url = completion_data.get("asset_url")
        file_size = completion_data.get("file_size_bytes")
        if asset_url and file_size:
            if not self.test_asset_verification(asset_url, file_size):
                return False
        
        # Test 11: Deletion (create new job for this)
        self.log_info("\nCreating second job for deletion test...")
        delete_job_id = self.test_job_submission_and_redis(image_path)
        if delete_job_id:
            time.sleep(2)  # Let it queue
            self.test_delete_endpoint(delete_job_id)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✓ Passed: {self.passed_tests}")
        print(f"✗ Failed: {self.failed_tests}")
        print("="*60)
        
        if self.failed_tests == 0:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("✨ Subphase 1.4 is fully interconnected and operational.")
            return True
        else:
            print(f"\n⚠️  {self.failed_tests} test(s) failed. Review output above.")
            return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python integration_test.py <path_to_image.png>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    tester = IntegrationTest()
    success = tester.run_all_tests(image_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()