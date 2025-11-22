"""
Updated Experiment model - Core experiment tracking with comparative evaluation support.

CHANGES FROM ORIGINAL:
1. Added: has_ground_truth field
2. Added: generate_baseline field (for user choice)
3. Updated: status field with more granular states
4. Added: variants relationship (ModelVariant)
5. Added: metrics relationship (ComparativeMetrics)
6. Removed: baseline_metrics relationship (replaced by metrics)
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base


class Experiment(Base):
    """
    Represents a comparative quantization experiment.
    
    Workflow:
    1. User creates experiment with dataset
    2. System detects if ground truth exists
    3. User chooses whether to generate baseline (if GT exists)
    4. User selects quantized models to compare
    5. System generates outputs from all variants
    6. System evaluates and displays comparison
    """
    __tablename__ = "experiments"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic info
    name = Column(String(255), nullable=False)
    # User-friendly name like "test2", "production_eval"
    
    baseline_model_id = Column(
        Integer,
        ForeignKey("models.id"),
        nullable=False
    )
    # Which model to use for baseline generation
    # Example: Llama-2-7B, Mistral-7B
    
    # Dataset information
    has_ground_truth = Column(Boolean, default=False)
    # True if uploaded dataset contains expected outputs
    # Determines available evaluation metrics
    
    sample_count = Column(Integer, default=0)
    # Number of samples in the uploaded dataset
    
    # User choices
    generate_baseline = Column(Boolean, nullable=True)
    # User's choice on baseline generation:
    # - null: Not yet decided (or not applicable)
    # - True: Generate baseline via Groq API
    # - False: Skip baseline, use only ground truth
    # Only relevant when has_ground_truth=True
    
    # Status tracking
    status = Column(String(50), default="created")
    # Possible values:
    # - 'created'                    → Just created, dataset not uploaded yet
    # - 'processing_dataset'         → Parsing uploaded JSON file
    # - 'awaiting_baseline_choice'   → User needs to choose baseline generation (if GT exists)
    # - 'generating_baseline'        → Generating baseline outputs via Groq
    # - 'awaiting_quantization'      → Ready for user to select quantized models
    # - 'generating_quantized'       → Generating quantized outputs
    # - 'ready_for_evaluation'       → All outputs generated, can evaluate
    # - 'evaluating'                 → Running comparative evaluation
    # - 'completed'                  → Evaluation complete, results available
    # - 'failed'                     → Error occurred during process
    is_draft = Column(Boolean, default=True)  # ← ADD THIS

    progress = Column(Integer, default=0)
    # Overall progress percentage: 0-100
    # Used for UI progress bar
    
    error_message = Column(String(1024), nullable=True)
    # If status='failed', stores detailed error message
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    model = relationship(
        "Model",
        backref="experiments"
    )
    
    samples = relationship(
        "DatasetSample",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="DatasetSample.position"
    )
    
    variants = relationship(
        "ModelVariant",
        back_populates="experiment",
        cascade="all, delete-orphan"
    )
    
    metrics = relationship(
        "ComparativeMetrics",
        back_populates="experiment",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return (
            f"<Experiment(id={self.id}, "
            f"name='{self.name}', "
            f"samples={self.sample_count}, "
            f"has_gt={self.has_ground_truth}, "
            f"status='{self.status}')>"
        )
    
    @property
    def is_complete(self):
        """Check if experiment is fully complete"""
        return self.status == 'completed'
    
    @property
    def has_baseline_variant(self):
        """Check if experiment has a baseline variant"""
        return any(v.variant_type == 'baseline' for v in self.variants)
    
    @property
    def quantized_variants(self):
        """Get list of quantized model variants"""
        return [v for v in self.variants if v.variant_type == 'quantized']
    
    @property
    def baseline_variant(self):
        """Get the baseline variant (if exists)"""
        for variant in self.variants:
            if variant.variant_type == 'baseline':
                return variant
        return None
