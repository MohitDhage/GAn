"""
==============================================================================
Subphase 1.3 — Celery Worker Entry Point
3D-GAN Generation System

File : worker.py

Purpose
-------
Celery discovers tasks by importing this module via the -A flag:
    celery -A worker worker --loglevel=info -Q gan3d.generate

Import order (no circular dependency):
    worker.py
        └── celery_app.py   (defines celery_app, redis_client)
        └── tasks.py        (imports celery_app; registers tasks onto it)

celery_app.py must NOT import tasks.py directly — that creates a cycle.
This thin entry-point module is the correct place to wire them together.
==============================================================================
"""

from celery_app import celery_app   # noqa: F401 — re-exported for -A discovery
import tasks                        # noqa: F401 — registers all tasks with celery_app
