"""
ModelVariant model - Represents each model version being compared in an experiment.

One experiment can have multiple variants:
- 1 baseline (optional if ground truth exists)
- 1-3 quantized versions (user selects)
- 0-1 bonus model (optional)
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base


class ModelVariant(Base):
    """
    Tracks each model version in a comparative experiment.
    
    Example variants for one experiment:
    - variant_type='baseline', model_name='llama-2-7b', quantization_level=None
    - variant_type='quantized', model_name='llama-2-7b', quantization_level='INT8'
    - variant_type='quantized', model_name='llama-2-7b', quantization_level='INT4'
    - variant_type='bonus', model_name='qwen-7b', quantization_level=None
    """
    __tablename__ = "model_variants"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    experiment_id = Column(
        Integer, 
        ForeignKey("experiments.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Model identification
    variant_type = Column(String(50), nullable=False)
    # Possible values: 'baseline', 'quantized', 'bonus'
    
    model_name = Column(String(255), nullable=False)
    # Examples: 'llama-2-7b', 'mistral-7b', 'qwen-7b', 'gpt-2'
    
    quantization_level = Column(String(50), nullable=True)
    # Values: None (for baseline/bonus), 'FP16', 'INT8', 'INT4'
    # This determines which GGUF file to load
    
    # Model configuration
    model_path = Column(String(512), nullable=True)
    # For quantized models: path to GGUF file
    # For baseline: null (uses Groq API)
    # For bonus: path or API identifier
    # Example: "/data/models/quantized/llama-2-7b-chat.Q8_0.gguf"
    
    inference_provider = Column(String(50), nullable=False, default='gguf')
    # Values: 'groq', 'gguf', 'api'
    # Determines how to generate outputs for this variant
    
    # Generation parameters
    generation_params = Column(JSON, nullable=True)
    # Stores temperature, max_tokens, top_p, etc.
    # Example: {"temperature": 0.7, "max_tokens": 256, "top_p": 0.9}
    
    # Status tracking
    status = Column(String(50), default='pending')
    # Possible values:
    # - 'pending'      → Created but not started
    # - 'generating'   → Currently generating outputs
    # - 'completed'    → All outputs generated
    # - 'failed'       → Error during generation
    # - 'evaluating'   → Generating comparative metrics
    # - 'cancelled'    → Generation was cancelled by user
    
    progress = Column(Float, default=0.0)
    # Generation progress: 0.0 to 1.0
    # Used for real-time progress tracking in UI
    
    celery_task_id = Column(String(255), nullable=True)
    # Celery task ID for this variant's generation task
    # Used to cancel/revoke running tasks
    
    error_message = Column(String(1024), nullable=True)
    # If status='failed', stores error details
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    experiment = relationship(
        "Experiment", 
        back_populates="variants"
    )
    
    generated_outputs = relationship(
        "GeneratedOutput",
        back_populates="variant",
        cascade="all, delete-orphan"
    )
    
    metrics = relationship(
        "ComparativeMetrics",
        back_populates="variant",
        cascade="all, delete-orphan",
        uselist=False  # One-to-one relationship
    )
    
    def __repr__(self):
        return (
            f"<ModelVariant(id={self.id}, "
            f"experiment_id={self.experiment_id}, "
            f"type={self.variant_type}, "
            f"model={self.model_name}, "
            f"quantization={self.quantization_level}, "
            f"status={self.status})>"
        )
    
    @property
    def display_name(self):
        """User-friendly display name for UI"""
        if self.variant_type == 'baseline':
            return f"{self.model_name} (Baseline)"
        elif self.variant_type == 'quantized':
            return f"{self.model_name} ({self.quantization_level})"
        else:  # bonus
            return f"{self.model_name} (Bonus)"
    
    @property
    def is_complete(self):
        """Check if all outputs have been generated"""
        return self.status == 'completed'
