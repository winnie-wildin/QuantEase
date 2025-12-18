"""
Celery application configuration for QuantEase.
Handles async tasks like model generation and evaluation.

For development/testing without Redis, set CELERY_ALWAYS_EAGER=true in environment.
By default, tasks run asynchronously via Redis broker for parallel processing.
"""
from celery import Celery
import os

# Get broker URL from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Eager mode: Only enable for testing without Redis (tasks run synchronously)
# Set CELERY_ALWAYS_EAGER=true to enable, otherwise defaults to False (async mode)
CELERY_ALWAYS_EAGER = os.getenv("CELERY_ALWAYS_EAGER", "false").lower() == "true"

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
    
    # Async execution: Set to False for production (uses Redis broker)
    # Set to True only for testing without Redis (tasks run synchronously in same process)
    task_always_eager=CELERY_ALWAYS_EAGER,
    task_eager_propagates=True,  # Propagate exceptions in eager mode
    
    # Worker configuration for parallel processing
    worker_prefetch_multiplier=1,  # Prefetch only 1 task per worker (better for long-running tasks)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevents memory leaks)
    task_acks_late=True,  # Acknowledge tasks after completion (better for long-running tasks)
    task_reject_on_worker_lost=True,  # Re-queue tasks if worker dies
)

# Auto-discover tasks - imports all tasks from the tasks package
celery_app.autodiscover_tasks(['app.tasks'])

# Explicitly import task modules to ensure they're registered
# This ensures tasks are available even if autodiscover has issues
from app.tasks import baseline_generation  # noqa: F401
from app.tasks import quantized_generation  # noqa: F401
from app.tasks import evaluation_task  # noqa: F401