"""
Similarity Calculator - Cosine similarity and embeddings
Uses sentence transformers for semantic similarity
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np


class SimilarityCalculator:
    """
    Calculate semantic similarity between texts using sentence embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize similarity calculator.
        
        Args:
            model_name: Sentence transformer model to use
                - all-MiniLM-L6-v2: Fast, good quality (default)
                - all-mpnet-base-v2: Better quality, slower
                - paraphrase-MiniLM-L6-v2: Optimized for paraphrase detection
        """
        self.model = SentenceTransformer(model_name)
    
    def calculate_cosine_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> List[float]:
        """
        Calculate cosine similarity between pairs of texts.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts (same length as texts1)
        
        Returns:
            List of similarity scores (0-1, higher is more similar)
        """
        if len(texts1) != len(texts2):
            raise ValueError("Both text lists must have same length")
        
        # Generate embeddings
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        
        # Calculate cosine similarity for each pair
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(float(sim))
        
        return similarities
    
    def calculate_average_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> float:
        """
        Calculate average cosine similarity across all pairs.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
        
        Returns:
            Average similarity score (0-1)
        """
        similarities = self.calculate_cosine_similarity(texts1, texts2)
        return float(np.mean(similarities))
    
    def calculate_single(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity for a single pair.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        similarities = self.calculate_cosine_similarity([text1], [text2])
        return similarities[0]


def calculate_output_divergence(
    baseline_outputs: List[str],
    quantized_outputs: List[str]
) -> float:
    """
    Calculate how much quantized outputs diverge from baseline.
    
    Returns:
        Divergence score (0-1, lower means more similar)
    """
    calculator = SimilarityCalculator()
    similarity = calculator.calculate_average_similarity(
        baseline_outputs,
        quantized_outputs
    )
    # Convert similarity to divergence
    divergence = 1.0 - similarity
    return divergence