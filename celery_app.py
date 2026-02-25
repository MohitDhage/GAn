"""
Celery Application Configuration
Ensures tasks are properly imported and registered
"""
import os
from celery import Celery

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB_BROKER = os.getenv("REDIS_DB_BROKER", "0")
REDIS_DB_BACKEND = os.getenv("REDIS_DB_BACKEND", "1")

# Celery app initialization
celery_app = Celery(
    "gan3d",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BROKER}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BACKEND}",
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Queue settings
    task_default_queue="gan3d.generate",
    task_default_exchange="gan3d.generate",
    task_default_routing_key="gan3d.generate",
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (important for GPU)
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
)

# Use Celery's autodiscovery — safe here because when the worker CLI
# loads this module, celery_app is fully initialised before tasks.py is
# imported.  The "tasks" string matches the tasks.py module in this directory.
celery_app.autodiscover_tasks(["tasks"], force=True)

if __name__ == "__main__":
    celery_app.start()