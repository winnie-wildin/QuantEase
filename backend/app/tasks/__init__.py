"""
Tasks package for QuantEase.
This package contains all Celery task definitions.
"""
from app.tasks.celery_app import celery_app

# Import tasks to ensure they're registered with Celery
from app.tasks import baseline_generation  # noqa: F401
from app.tasks import quantized_generation  # noqa: F401
from app.tasks import evaluation_task  # noqa: F401

__all__ = ['celery_app']
