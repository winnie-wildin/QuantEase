#app/models/comparative_metrics.py
"""
ComparativeMetrics model - Stores evaluation results for each model variant.

Contains three types of metrics:
1. Independent metrics: size, latency, repetition (don't require comparison)
2. Comparative vs ground truth: BERTScore, cosine similarity, perplexity
3. Comparative vs baseline: BERTScore, cosine similarity (quality preservation)
"""

from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base


class ComparativeMetrics(Base):
    """
    Stores comprehensive evaluation metrics for a model variant.
    
    Calculated after all outputs are generated for the variant.
    One record per variant per experiment.
    """
    __tablename__ = "comparative_metrics"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    variant_id = Column(
        Integer,
        ForeignKey("model_variants.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One-to-one relationship
        index=True
    )
    
    experiment_id = Column(
        Integer,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # ===== INDEPENDENT METRICS =====
    # These don't require comparison with other outputs
    
    model_size_mb = Column(Float, nullable=True)
    # Size of the model file in megabytes
    # For GGUF files: extracted from file size
    # For API models: estimated or null
    
    avg_latency_ms = Column(Float, nullable=True)
    # Average generation latency across all samples
    # Lower is better
    
    avg_token_count = Column(Float, nullable=True)
    # Average number of tokens generated per sample
    # Neither good nor bad, just informative
    
    avg_tokens_per_second = Column(Float, nullable=True)
    # Average generation speed
    # Higher is better
    
    avg_repetition_score = Column(Float, nullable=True)
    # Average repetition across all outputs
    # 0.0 = no repetition, 1.0 = highly repetitive
    # Lower is better
    
    # ===== COMPARATIVE METRICS VS GROUND TRUTH =====
    # Only calculated if experiment has ground truth
    
    bertscore_precision_vs_gt = Column(Float, nullable=True)
    # BERTScore precision against ground truth
    # 0.0 to 1.0, higher is better
    
    bertscore_recall_vs_gt = Column(Float, nullable=True)
    # BERTScore recall against ground truth
    # 0.0 to 1.0, higher is better
    
    bertscore_f1_vs_gt = Column(Float, nullable=True)
    # BERTScore F1 (harmonic mean) - PRIMARY METRIC
    # 0.0 to 1.0, higher is better
    
    cosine_similarity_vs_gt = Column(Float, nullable=True)
    # Sentence embedding cosine similarity vs ground truth
    # 0.0 to 1.0, higher is better
    
    avg_perplexity_vs_gt = Column(Float, nullable=True)
    # Average perplexity when evaluating ground truth
    # Lower is better
    # Only meaningful when comparing to ground truth
    
    # ===== COMPARATIVE METRICS VS BASELINE =====
    # Only calculated if experiment has baseline variant
    
    bertscore_precision_vs_baseline = Column(Float, nullable=True)
    # BERTScore precision against baseline outputs
    # 0.0 to 1.0, higher is better
    
    bertscore_recall_vs_baseline = Column(Float, nullable=True)
    # BERTScore recall against baseline outputs
    # 0.0 to 1.0, higher is better
    
    bertscore_f1_vs_baseline = Column(Float, nullable=True)
    # BERTScore F1 vs baseline - KEY METRIC for quality preservation
    # 0.0 to 1.0, higher is better
    # Goal: Keep > 0.9 after quantization
    
    cosine_similarity_vs_baseline = Column(Float, nullable=True)
    # Sentence embedding similarity vs baseline outputs
    # 0.0 to 1.0, higher is better
    
    output_divergence_score = Column(Float, nullable=True)
    # Custom metric: how different outputs are from baseline
    # 0.0 = identical, 1.0 = completely different
    # Lower is better
    
    # ===== METADATA =====
    
    evaluation_status = Column(String(50), default='pending')
    # Possible values:
    # - 'pending'    → Not evaluated yet
    # - 'evaluating' → Currently calculating metrics
    # - 'completed'  → All metrics calculated
    # - 'failed'     → Error during evaluation
    
    evaluation_error = Column(String(1024), nullable=True)
    # If evaluation failed, store error message
    
    samples_evaluated = Column(Integer, default=0)
    # Number of samples successfully evaluated
    # Should match experiment.sample_count when complete

    evaluation_results = Column(JSON, nullable=True)
    # Stores task-specific evaluation results as JSON
    # Schema varies by task type:
    # - text_generation: {"bertscore_f1": 87.5, "length_ratio": 0.98, ...}
    # - classification: {"accuracy": 92.3, "macro_f1": 89.1, ...}
    # - rag: {"answer_relevance": 85.2, "llm_judge": {...}, ...}
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    variant = relationship(
        "ModelVariant",
        back_populates="metrics"
    )
    
    experiment = relationship(
        "Experiment",
        back_populates="metrics"
    )
    
    def __repr__(self):
        return (
            f"<ComparativeMetrics(id={self.id}, "
            f"variant_id={self.variant_id}, "
            f"size={self.model_size_mb}MB, "
            f"latency={self.avg_latency_ms}ms, "
            f"bert_vs_gt={self.bertscore_f1_vs_gt}, "
            f"bert_vs_base={self.bertscore_f1_vs_baseline})>"
        )
    
    @property
    def has_ground_truth_metrics(self):
        """Check if metrics vs ground truth are available"""
        return self.bertscore_f1_vs_gt is not None
    
    @property
    def has_baseline_metrics(self):
        """Check if metrics vs baseline are available"""
        return self.bertscore_f1_vs_baseline is not None
    
    @property
    def quality_score(self):
        """
        Calculate overall quality score (0-100)
        Combines BERTScore vs GT (if available) with other metrics
        """
        if self.bertscore_f1_vs_gt:
            return round(self.bertscore_f1_vs_gt * 100, 2)
        elif self.bertscore_f1_vs_baseline:
            return round(self.bertscore_f1_vs_baseline * 100, 2)
        return None
    
    @property
    def efficiency_score(self):
        """
        Calculate efficiency score based on size and speed
        Lower size + higher speed = better efficiency
        """
        if not self.model_size_mb or not self.avg_latency_ms:
            return None
        
        # Normalize: smaller size and lower latency = better
        # This is a simple heuristic, can be refined
        size_factor = 1.0 / (self.model_size_mb / 1000.0)  # Normalize to GB
        speed_factor = 1.0 / (self.avg_latency_ms / 1000.0)  # Normalize to seconds
        
        return round((size_factor + speed_factor) / 2 * 100, 2)
