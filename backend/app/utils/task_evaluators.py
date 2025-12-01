#app/utils/task_evaluators.py
"""
Task-specific evaluation logic for Text Generation, Classification, and RAG.
Each evaluator computes appropriate metrics for its task type.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from collections import Counter
from app.utils.bertscore_calculator import BERTScoreCalculator


class TextGenEvaluator:
    """
    Evaluator for Text Generation tasks.
    Metrics: BERTScore, Length Ratio, Divergence
    """
    
    def __init__(self):
        self.bert_calc = BERTScoreCalculator()
    
    def evaluate(
        self, 
        candidates: List[str], 
        references: List[str],
        baseline_outputs: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate text generation quality.
        
        Args:
            candidates: Generated outputs from quantized model
            references: Ground truth or baseline outputs
            baseline_outputs: Optional baseline outputs for divergence calculation
        
        Returns:
            Dict with all text generation metrics
        """
        results = {}
        
        # 1. BERTScore (semantic similarity to reference)
        print("   📊 Computing BERTScore...")
        bert_scores = self.bert_calc.calculate(candidates, references)
        results["bertscore_f1"] = bert_scores["f1"]  # ✅ Keep as 0-1 scale
        results["bertscore_precision"] = bert_scores["precision"]  # ✅
        results["bertscore_recall"] = bert_scores["recall"]  # ✅
        
        # 2. Length Ratio (sanity check for hallucination)
        print("   📏 Computing Length Ratio...")
        length_ratios = []
        for cand, ref in zip(candidates, references):
            ref_len = len(ref.split())
            cand_len = len(cand.split())
            ratio = cand_len / ref_len if ref_len > 0 else 1.0
            length_ratios.append(ratio)
        
        results["length_ratio_mean"] = float(np.mean(length_ratios))
        results["length_ratio_std"] = float(np.std(length_ratios))
        
        # 3. Divergence (if baseline provided)
        if baseline_outputs and len(baseline_outputs) == len(candidates):
            print("   🔀 Computing Divergence from baseline...")
            divergence_scores = self.bert_calc.calculate(candidates, baseline_outputs)
            # Divergence = 1 - similarity (how much it differs from baseline)
            results["divergence_score"] = 1 - divergence_scores["f1"]  # ✅ Keep as 0-1 scale
        else:
            results["divergence_score"] = None
        
        return results


class ClassificationEvaluator:
    """
    Evaluator for Classification tasks.
    Metrics: Accuracy, Macro F1, Weighted F1, Per-class F1, Confusion Matrix
    """
    
    def detect_imbalance(self, labels: List[str], threshold: float = 2.0) -> Tuple[bool, Dict]:
        """
        Detect class imbalance in the dataset.
        
        Args:
            labels: All true labels
            threshold: Ratio threshold (default 2:1)
        
        Returns:
            (is_imbalanced, stats_dict)
        """
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        
        ratio = max_count / min_count if min_count > 0 else float('inf')
        is_imbalanced = ratio > threshold
        
        return is_imbalanced, {
            "class_distribution": dict(label_counts),
            "imbalance_ratio": ratio,
            "most_common_class": label_counts.most_common(1)[0][0],
            "least_common_class": label_counts.most_common()[-1][0]
        }
    
    def evaluate(
        self,
        predictions: List[str],
        true_labels: List[str]
    ) -> Dict:
        """
        Evaluate classification performance.
        
        Args:
            predictions: Predicted labels from model
            true_labels: Ground truth labels
        
        Returns:
            Dict with all classification metrics
        """
        results = {}
        
        # Get unique classes
        unique_labels = sorted(list(set(true_labels + predictions)))
        num_classes = len(unique_labels)
        
        # Detect class imbalance
        is_imbalanced, imbalance_stats = self.detect_imbalance(true_labels)
        results["is_imbalanced"] = is_imbalanced
        results["imbalance_stats"] = imbalance_stats
        
        # 1. Accuracy
        print("   🎯 Computing Accuracy...")
        results["accuracy"] = accuracy_score(true_labels, predictions)
        
        # 2. Macro F1 (primary metric - fair across classes)
        print("   📊 Computing Macro F1...")
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, 
            predictions,
            average='macro',
            zero_division=0
        )
        results["macro_precision"] = precision
        results["macro_recall"] = recall
        results["macro_f1"] = f1
        
        # 3. Weighted F1 (if imbalanced)
        if is_imbalanced:
            print("   ⚖️ Computing Weighted F1 (dataset is imbalanced)...")
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                true_labels,
                predictions,
                average='weighted',
                zero_division=0
            )
            results["weighted_precision"] = precision_w
            results["weighted_recall"] = recall_w
            results["weighted_f1"] = f1_w
                    
        # 4. Per-class metrics
        print("   📋 Computing per-class metrics...")
        per_class_metrics = precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=unique_labels,
            zero_division=0
        )
        
        results["per_class_f1"] = {
            label: float(f1_score)
            for label, f1_score in zip(unique_labels, per_class_metrics[2])
        }
                
        # 5. Confusion Matrix
        print("   🔢 Computing confusion matrix...")
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        results["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": unique_labels
        }
        
        # 6. Binary classification special handling
        if num_classes == 2:
            print("   ✅ Binary classification detected - adding binary metrics...")
            # For binary, also compute standard binary F1
            binary_f1 = f1_score_binary(true_labels, predictions, pos_label=unique_labels[1])
            results["binary_f1"] = binary_f1
            results["is_binary"] = True
        else:
            results["is_binary"] = False
        
        results["num_classes"] = num_classes
        
        return results


class RAGEvaluator:
    """
    Evaluator for Retrieval/Factual QA tasks.
    Metrics: Answer Relevance, (Optional) LLM Judge for Hallucination & Correctness
    """
    
    def __init__(self):
        self.bert_calc = BERTScoreCalculator()
    
    def evaluate(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict:
        """
        Evaluate RAG/QA performance.
        
        Args:
            generated_answers: Answers from model
            reference_answers: Ground truth answers
        
        Returns:
            Dict with RAG metrics
        """
        results = {}
        
        # 1. Answer Relevance (using embeddings)
        print("   🔍 Computing Answer Relevance...")
        relevance_scores = self.bert_calc.calculate(generated_answers, reference_answers)
        results["answer_relevance"] = relevance_scores["f1"] 
        results["relevance_precision"] = relevance_scores["precision"]
        results["relevance_recall"] = relevance_scores["recall"]
        
        # Store per-sample relevance for judge sampling
        results["per_sample_relevance"] = [s for s in relevance_scores["similarities"]]
        
        return results


def f1_score_binary(y_true: List[str], y_pred: List[str], pos_label: str) -> float:
    """Helper to calculate binary F1 score."""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)