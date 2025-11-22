"""
Updated DatasetSample model - Stores input samples and optional ground truth.

CHANGES FROM ORIGINAL:
1. Renamed: input → input_text (more explicit)
2. Renamed: expected_output → ground_truth_output (matches terminology)
3. Removed: generated_output field (moved to GeneratedOutput table)
4. Removed: is_selected field (not needed in new design)
5. Added: position field (for maintaining order)
6. Added: generated_outputs relationship (one-to-many)
"""

from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.base import Base


class DatasetSample(Base):
    """
    Represents one sample from the uploaded dataset.
    
    Contains:
    - Input text (required)
    - Ground truth output (optional, from uploaded JSON)
    - Position in dataset (for ordering)
    
    Generated outputs are stored separately in GeneratedOutput table,
    allowing multiple outputs per sample (one per model variant).
    """
    __tablename__ = "dataset_samples"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key
    experiment_id = Column(
        Integer,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Sample data
    input_text = Column(Text, nullable=False)
    # The input prompt/question from the dataset
    # Example: "What is artificial intelligence?"
    
    ground_truth_output = Column(Text, nullable=True)
    # Expected output from uploaded dataset (if provided)
    # Example: "Artificial intelligence is..."
    # Will be null if dataset only contains inputs
    
    position = Column(Integer, nullable=False)
    # Position in the dataset (0-indexed)
    # Used to maintain original order in UI display
    # Example: Sample #1, Sample #2, etc.
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    experiment = relationship(
        "Experiment",
        back_populates="samples"
    )
    
    generated_outputs = relationship(
        "GeneratedOutput",
        back_populates="sample",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        has_gt = "with GT" if self.ground_truth_output else "no GT"
        return (
            f"<DatasetSample(id={self.id}, "
            f"experiment_id={self.experiment_id}, "
            f"position={self.position}, "
            f"{has_gt})>"
        )
    
    @property
    def input_preview(self):
        """Return first 50 characters of input for display"""
        if not self.input_text:
            return ""
        return self.input_text[:50] + "..." if len(self.input_text) > 50 else self.input_text
    
    @property
    def has_ground_truth(self):
        """Check if this sample has ground truth output"""
        return self.ground_truth_output is not None and len(self.ground_truth_output) > 0
    
    def get_output_for_variant(self, variant_id):
        """
        Get the generated output for a specific model variant.
        
        Args:
            variant_id: ID of the ModelVariant
            
        Returns:
            GeneratedOutput object or None if not found
        """
        for output in self.generated_outputs:
            if output.variant_id == variant_id:
                return output
        return None
