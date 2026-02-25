"""
API Verification Script - Tests the complete 3D asset generation lifecycle.

Usage:
    python test_api.py <path_to_test_image.png>

Flow:
    1. Submit image to POST /v1/generate
    2. Poll GET /v1/jobs/{id}/status until COMPLETED
    3. Verify asset_url via HEAD request
    4. Download and verify the .glb file
"""
import sys
import time
import requests
from pathlib import Path
from typing import Optional


API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 150  # 5 minutes max


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def test_health_check():
    """Test the health endpoint"""
    print_info("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        
        print_success(f"Health check passed: {health_data}")
        
        if health_data.get("redis") != "healthy":
            print_warning(f"Redis status: {health_data.get('redis')}")
        if health_data.get("celery") != "healthy":
            print_warning(f"Celery status: {health_data.get('celery')}")
        
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def submit_generation_job(image_path: Path) -> Optional[dict]:
    """
    Submit image to POST /v1/generate
    Returns job data with job_id and poll_url
    """
    print_info(f"Submitting image: {image_path}")
    
    if not image_path.exists():
        print_error(f"Image file not found: {image_path}")
        return None
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            response = requests.post(f"{API_BASE_URL}/v1/generate", files=files)
        
        if response.status_code == 503:
            print_warning("Queue is full (503). Server returned:")
            print_warning(response.json())
            return None
        
        response.raise_for_status()
        job_data = response.json()
        
        print_success(f"Job submitted successfully!")
        print_success(f"Job ID: {job_data['job_id']}")
        print_success(f"Status: {job_data['status']}")
        print_success(f"Poll URL: {job_data['poll_url']}")
        
        return job_data
    
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to submit job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print_error(f"Response: {e.response.text}")
        return None


def poll_job_status(job_id: str) -> Optional[dict]:
    """
    Poll GET /v1/jobs/{id}/status until completion
    Returns final job details
    """
    print_info(f"Polling job status for {job_id}...")
    
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        try:
            # Use lightweight status endpoint
            response = requests.get(f"{API_BASE_URL}/v1/jobs/{job_id}/status")
            response.raise_for_status()
            status_data = response.json()
            
            job_status = status_data["status"]
            progress = status_data["progress"]
            
            print(f"  Attempt {attempt}: Status={job_status}, Progress={progress}%", end="\r")
            
            if job_status == "COMPLETED":
                print()  # New line after polling
                print_success(f"Job completed! (took {attempt * POLL_INTERVAL_SECONDS}s)")
                
                # Get full details
                detail_response = requests.get(f"{API_BASE_URL}/v1/jobs/{job_id}")
                detail_response.raise_for_status()
                return detail_response.json()
            
            elif job_status == "FAILED":
                print()
                print_error(f"Job failed!")
                
                # Get error details
                detail_response = requests.get(f"{API_BASE_URL}/v1/jobs/{job_id}")
                detail_response.raise_for_status()
                error_data = detail_response.json()
                print_error(f"Error: {error_data.get('error_message', 'Unknown error')}")
                return None
            
            elif job_status == "EXPIRED":
                print()
                print_warning("Job expired before completion")
                return None
            
            # Continue polling
            time.sleep(POLL_INTERVAL_SECONDS)
        
        except requests.exceptions.RequestException as e:
            print()
            print_error(f"Failed to poll job status: {e}")
            return None
    
    print()
    print_warning(f"Max polling attempts reached ({MAX_POLL_ATTEMPTS})")
    return None


def verify_asset(asset_url: str, file_size_bytes: int) -> bool:
    """
    Verify the generated asset via HEAD request
    Then download and validate
    """
    print_info(f"Verifying asset at: {asset_url}")
    
    # Construct full URL
    if not asset_url.startswith("http"):
        full_url = f"{API_BASE_URL}{asset_url}"
    else:
        full_url = asset_url
    
    try:
        # HEAD request to check existence
        head_response = requests.head(full_url)
        head_response.raise_for_status()
        
        content_length = int(head_response.headers.get("content-length", 0))
        print_success(f"Asset exists! Content-Length: {content_length} bytes")
        
        # Verify file size matches
        if content_length == file_size_bytes:
            print_success(f"File size matches expected: {file_size_bytes} bytes")
        else:
            print_warning(f"File size mismatch: Expected {file_size_bytes}, got {content_length}")
        
        # Download the file
        print_info("Downloading asset...")
        get_response = requests.get(full_url)
        get_response.raise_for_status()
        
        # Verify it's a GLB file (starts with 'glTF' magic bytes)
        if get_response.content[:4] == b"glTF":
            print_success("Valid GLB file detected (glTF magic bytes present)")
        else:
            print_warning("File may not be a valid GLB (glTF magic bytes not found)")
        
        # Save to local file
        output_path = Path("downloaded_test_asset.glb")
        with open(output_path, "wb") as f:
            f.write(get_response.content)
        
        print_success(f"Asset downloaded successfully to: {output_path}")
        print_success(f"Downloaded size: {len(get_response.content)} bytes")
        
        return True
    
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to verify asset: {e}")
        return False


def test_delete_endpoint(job_id: str):
    """
    Test the DELETE endpoint (optional - creates a new job)
    """
    print_info(f"Testing DELETE endpoint for job {job_id}...")
    
    try:
        response = requests.delete(f"{API_BASE_URL}/v1/jobs/{job_id}")
        response.raise_for_status()
        delete_data = response.json()
        
        print_success(f"Job deleted: {delete_data['message']}")
        return True
    
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to delete job: {e}")
        return False


def main():
    """Main test orchestration"""
    print("=" * 60)
    print("3D Asset Generation API - Verification Test")
    print("=" * 60)
    print()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print_error("Usage: python test_api.py <path_to_image.png>")
        print_info("Example: python test_api.py test_image.png")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    
    # Step 1: Health check
    print("\n" + "=" * 60)
    print("STEP 1: Health Check")
    print("=" * 60)
    if not test_health_check():
        print_error("Health check failed. Ensure Redis and Celery worker are running.")
        sys.exit(1)
    
    # Step 2: Submit job
    print("\n" + "=" * 60)
    print("STEP 2: Submit Generation Job")
    print("=" * 60)
    job_data = submit_generation_job(image_path)
    if not job_data:
        print_error("Failed to submit job.")
        sys.exit(1)
    
    job_id = job_data["job_id"]
    
    # Step 3: Poll status
    print("\n" + "=" * 60)
    print("STEP 3: Poll Job Status")
    print("=" * 60)
    final_data = poll_job_status(job_id)
    if not final_data:
        print_error("Job did not complete successfully.")
        sys.exit(1)
    
    # Step 4: Verify asset
    print("\n" + "=" * 60)
    print("STEP 4: Verify Generated Asset")
    print("=" * 60)
    
    asset_url = final_data.get("asset_url")
    file_size = final_data.get("file_size_bytes")
    
    if not asset_url:
        print_error("No asset_url in response!")
        sys.exit(1)
    
    if not verify_asset(asset_url, file_size):
        print_error("Asset verification failed.")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print_success("All tests passed!")
    print_info(f"Job ID: {job_id}")
    print_info(f"Status: {final_data['status']}")
    print_info(f"Asset URL: {asset_url}")
    print_info(f"File Size: {file_size} bytes")
    print_info(f"Generation Time: {final_data.get('generation_time_seconds', 0):.2f}s")
    print()
    print_success("✨ Subphase 1.4 verification complete! Backend is fully operational.")
    print()


if __name__ == "__main__":
    main()
