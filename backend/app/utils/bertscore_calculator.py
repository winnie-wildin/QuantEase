"""
BERTScore Calculator - Semantic similarity evaluation
Uses BERT embeddings to compare generated outputs with references
"""
from bert_score import score as bert_score
from typing import List, Dict, Tuple
import numpy as np


class BERTScoreCalculator:
    """
    Calculate BERTScore metrics for comparing model outputs.
    
    BERTScore measures semantic similarity using contextual embeddings.
    Returns precision, recall, and F1 scores between 0 and 1.
    """
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        """
        Initialize BERTScore calculator.
        
        Args:
            model_type: Pretrained model to use for embeddings
                - microsoft/deberta-xlarge-mnli (best, but slower)
                - microsoft/deberta-base-mnli (good balance)
                - bert-base-uncased (fastest)
        """
        self.model_type = model_type
    
    def calculate(
        self,
        candidates: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate BERTScore between candidates and references.
        
        Args:
            candidates: List of generated outputs to evaluate
            references: List of reference texts (ground truth or baseline)
        
        Returns:
            Dict with precision, recall, and F1 scores
        """
        if len(candidates) != len(references):
            raise ValueError("Candidates and references must have same length")
        
        if not candidates:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        
        # Calculate BERTScore
        P, R, F1 = bert_score(
            candidates,
            references,
            model_type=self.model_type,
            verbose=False
        )
        
        # Average scores
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item())
        }
    
    def calculate_single(
        self,
        candidate: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate BERTScore for a single pair.
        
        Args:
            candidate: Generated output
            reference: Reference text
        
        Returns:
            Dict with precision, recall, and F1 scores
        """
        return self.calculate([candidate], [reference])


def calculate_bertscore_batch(
    candidates: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-base-mnli"
) -> Dict[str, float]:
    """
    Convenience function to calculate BERTScore.
    
    Args:
        candidates: List of generated outputs
        references: List of reference texts
        model_type: Model to use (default: deberta-base for speed)
    
    Returns:
        Dict with average precision, recall, F1
    """
    calculator = BERTScoreCalculator(model_type=model_type)
    return calculator.calculate(candidates, references)