"""
Celery application configuration for QuantEase.
Handles async tasks like model generation and evaluation.
"""
from celery import Celery
import os

# Get broker URL from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "quantease",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    task_always_eager=True,  # Run tasks synchronously for testing (no Redis needed)
    task_eager_propagates=True,  # Propagate exceptions in eager mode
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['app.tasks'])