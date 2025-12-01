#app/utils/bertscore_calculator.py
"""
Optimized BERTScore calculator using all-mpnet-base-v2.
Replaces DeBERTa-base-mnli for better performance and multilingual support.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import torch


class BERTScoreCalculator:
    """
    Calculate semantic similarity using all-mpnet-base-v2 embeddings.
    This model is faster, smaller (420MB), and better for general-purpose tasks.
    """
    
    def __init__(self):
        """Initialize the embedding model."""
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"✅ BERTScore calculator initialized on {self.device}")
    
    def calculate(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore-style metrics using cosine similarity.
        
        Args:
            candidates: List of generated outputs
            references: List of reference outputs (ground truth or baseline)
        
        Returns:
            Dict with precision, recall, and F1 scores (0-1 scale)
        """
        if len(candidates) != len(references):
            raise ValueError(f"Mismatch: {len(candidates)} candidates vs {len(references)} references")
        
        if not candidates or not references:
            raise ValueError("Empty input lists")
        
        # Encode all texts in batches for efficiency
        print(f"   Encoding {len(candidates)} candidate-reference pairs...")
        candidate_embeddings = self.model.encode(
            candidates, 
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        reference_embeddings = self.model.encode(
            references,
            batch_size=32, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Calculate pairwise cosine similarities
        similarities = []
        for cand_emb, ref_emb in zip(candidate_embeddings, reference_embeddings):
            # Cosine similarity: dot product of normalized vectors
            cand_norm = cand_emb / np.linalg.norm(cand_emb)
            ref_norm = ref_emb / np.linalg.norm(ref_emb)
            sim = np.dot(cand_norm, ref_norm)
            similarities.append(float(sim))
        
        # Convert to percentage (0-100%)
        avg_similarity = np.mean(similarities)
        
        # BERTScore-style metrics (in this simplified version, P=R=F1)
        # For token-level BERTScore, P≠R, but for sentence embeddings they're equivalent
        return {
            "precision": avg_similarity,
            "recall": avg_similarity,
            "f1": avg_similarity,
            "similarities": similarities  # Per-sample scores
        }
    
    def calculate_single(self, candidate: str, reference: str) -> float:
        """
        Calculate similarity for a single candidate-reference pair.
        
        Args:
            candidate: Generated output
            reference: Reference output
            
        Returns:
            Similarity score (0-1)
        """
        cand_emb = self.model.encode([candidate], convert_to_numpy=True)[0]
        ref_emb = self.model.encode([reference], convert_to_numpy=True)[0]
        
        cand_norm = cand_emb / np.linalg.norm(cand_emb)
        ref_norm = ref_emb / np.linalg.norm(ref_emb)
        
        return float(np.dot(cand_norm, ref_norm))