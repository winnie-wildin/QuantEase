"""
GeneratedOutput model - Stores generated outputs from each model variant.

Each dataset sample will have multiple outputs:
- One from each model variant (baseline, quantized versions, bonus)
- Allows side-by-side comparison in the dashboard
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base


class GeneratedOutput(Base):
    """
    Stores the generated output for a specific dataset sample from a specific model variant.
    
    Example: 
    - Sample #1, Baseline variant → "AI is artificial intelligence..."
    - Sample #1, INT8 variant → "AI is artificial intelligence..."
    - Sample #1, INT4 variant → "Artificial intelligence is..."
    """
    __tablename__ = "generated_outputs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    sample_id = Column(
        Integer,
        ForeignKey("dataset_samples.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    variant_id = Column(
        Integer,
        ForeignKey("model_variants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Generated content
    output_text = Column(Text, nullable=False)
    # The actual generated text from the model
    
    # Performance metrics (measured during generation)
    latency_ms = Column(Float, nullable=True)
    # Time taken to generate this output in milliseconds
    
    token_count = Column(Integer, nullable=True)
    # Number of tokens in the generated output
    
    tokens_per_second = Column(Float, nullable=True)
    # Generation speed: tokens / (latency_ms / 1000)
    
    # Quality metrics (calculated after generation)
    repetition_score = Column(Float, nullable=True)
    # Measures n-gram repetition: 0.0 (no repetition) to 1.0 (highly repetitive)
    # Lower is better
    
    # Generation metadata
    generation_params_used = Column(JSON, nullable=True)
    # Actual parameters used for this generation
    # May differ from variant's default params
    # Example: {"temperature": 0.7, "max_tokens": 256, "seed": 42}
    
    prompt_template = Column(Text, nullable=True)
    # The complete prompt sent to the model (for debugging)
    # Includes system message + input formatting
    
    # Error handling
    generation_error = Column(String(1024), nullable=True)
    # If generation failed for this sample, store error message
    
    is_successful = Column(
        Integer, 
        default=1,
        nullable=False
    )
    # 1 = successfully generated, 0 = generation failed
    # Using Integer instead of Boolean for better SQLite compatibility
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    sample = relationship(
        "DatasetSample",
        back_populates="generated_outputs"
    )
    
    variant = relationship(
        "ModelVariant",
        back_populates="generated_outputs"
    )
    
    def __repr__(self):
        return (
            f"<GeneratedOutput(id={self.id}, "
            f"sample_id={self.sample_id}, "
            f"variant_id={self.variant_id}, "
            f"tokens={self.token_count}, "
            f"latency={self.latency_ms}ms)>"
        )
    
    @property
    def preview(self):
        """Return first 100 characters of output for display"""
        if not self.output_text:
            return ""
        return self.output_text[:100] + "..." if len(self.output_text) > 100 else self.output_text
    
    @property
    def tokens_per_second_calculated(self):
        """Calculate tokens per second if not already stored"""
        if self.tokens_per_second:
            return self.tokens_per_second
        if self.token_count and self.latency_ms and self.latency_ms > 0:
            return self.token_count / (self.latency_ms / 1000.0)
        return None
