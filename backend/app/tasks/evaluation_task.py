#app/tasks/evaluation_task.py
"""
Celery task for evaluating model variants.
Calculates BERTScore, cosine similarity, and performance metrics.
"""
from app.tasks.celery_app import celery_app
from app.database import SessionLocal
from app.models import (
    Experiment, ModelVariant, GeneratedOutput, 
    ComparativeMetrics, DatasetSample
)
from app.utils.bertscore_calculator import BERTScoreCalculator
from app.utils.similarity_calculator import SimilarityCalculator
from sqlalchemy.orm import Session
import numpy as np


@celery_app.task(bind=True, name="evaluate_variant")
def evaluate_variant(self, variant_id: int):
    """
    Evaluate a model variant by calculating comprehensive metrics.
    
    Args:
        variant_id: ID of the variant to evaluate
    
    Returns:
        Dict with evaluation results
    """
    db = SessionLocal()
    
    try:
        # Get variant and experiment
        variant = db.query(ModelVariant).filter(ModelVariant.id == variant_id).first()
        if not variant:
            raise ValueError(f"Variant {variant_id} not found")
        
        experiment = variant.experiment
        
        # Get all outputs for this variant
        outputs = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == variant_id,
            GeneratedOutput.is_successful == 1
        ).all()
        
        if not outputs:
            raise ValueError(f"No successful outputs found for variant {variant_id}")
        
        print(f"📊 Evaluating variant {variant_id} ({len(outputs)} outputs)...")
        
        # Initialize calculators
        bert_calc = BERTScoreCalculator(model_type="microsoft/deberta-base-mnli")
        sim_calc = SimilarityCalculator()
        
        # Create or get metrics record
        metrics = db.query(ComparativeMetrics).filter(
            ComparativeMetrics.variant_id == variant_id
        ).first()
        
        if not metrics:
            metrics = ComparativeMetrics(
                variant_id=variant_id,
                experiment_id=experiment.id
            )
            db.add(metrics)
        
        metrics.evaluation_status = "running"
        db.commit()
        
        # ===== CALCULATE PERFORMANCE METRICS =====
        latencies = [o.latency_ms for o in outputs if o.latency_ms]
        token_counts = [o.token_count for o in outputs if o.token_count]
        speeds = [o.tokens_per_second for o in outputs if o.tokens_per_second]
        
        metrics.avg_latency_ms = float(np.mean(latencies)) if latencies else None
        metrics.avg_token_count = float(np.mean(token_counts)) if token_counts else None
        metrics.avg_tokens_per_second = float(np.mean(speeds)) if speeds else None
        
        # Model size (if GGUF)
        if variant.model_path:
            import os
            if os.path.exists(variant.model_path):
                metrics.model_size_mb = os.path.getsize(variant.model_path) / (1024 * 1024)
        
        # ===== SMART EVALUATION LOGIC =====
        # Determine reference: ground truth if available, otherwise baseline
        
        if experiment.has_ground_truth:
            print("   ✅ Using ground truth as reference...")
            # Get ground truth texts
            ground_truths = []
            candidate_texts = []
            
            for output in outputs:
                sample = output.sample
                if sample.ground_truth_output:
                    ground_truths.append(sample.ground_truth_output)
                    candidate_texts.append(output.output_text)
            
            if ground_truths:
                # Evaluate against ground truth
                bert_scores = bert_calc.calculate(candidate_texts, ground_truths)
                metrics.bertscore_precision_vs_gt = bert_scores["precision"]
                metrics.bertscore_recall_vs_gt = bert_scores["recall"]
                metrics.bertscore_f1_vs_gt = bert_scores["f1"]
                
                metrics.cosine_similarity_vs_gt = sim_calc.calculate_average_similarity(
                    candidate_texts, ground_truths
                )
        
        else:
            print("   ℹ️  No ground truth - using baseline as reference...")
            # Find baseline variant
            baseline = db.query(ModelVariant).filter(
                ModelVariant.experiment_id == experiment.id,
                ModelVariant.variant_type == "baseline"
            ).first()
            
            if baseline and variant.id != baseline.id:
                # This is a quantized/bonus variant - compare to baseline
                baseline_outputs = db.query(GeneratedOutput).filter(
                    GeneratedOutput.variant_id == baseline.id,
                    GeneratedOutput.is_successful == 1
                ).all()
                
                if baseline_outputs:
                    # Match outputs by sample_id
                    baseline_texts = []
                    candidate_texts = []
                    
                    for output in outputs:
                        baseline_out = next(
                            (b for b in baseline_outputs if b.sample_id == output.sample_id),
                            None
                        )
                        if baseline_out:
                            baseline_texts.append(baseline_out.output_text)
                            candidate_texts.append(output.output_text)
                    
                    if baseline_texts:
                        # Use baseline as "ground truth" for variants
                        bert_scores = bert_calc.calculate(candidate_texts, baseline_texts)
                        metrics.bertscore_precision_vs_gt = bert_scores["precision"]
                        metrics.bertscore_recall_vs_gt = bert_scores["recall"]
                        metrics.bertscore_f1_vs_gt = bert_scores["f1"]
                        
                        metrics.cosine_similarity_vs_gt = sim_calc.calculate_average_similarity(
                            candidate_texts, baseline_texts
                        )
                        
                        print(f"   ✅ Compared against baseline: BERTScore F1 = {metrics.bertscore_f1_vs_gt:.4f}")
            
            elif variant.variant_type == "baseline":
                # This IS the baseline - no reference to compare against
                print("   ℹ️  Baseline variant - no quality comparison needed")
        
        # ===== ALWAYS COMPARE QUANTIZED VS BASELINE =====
        if variant.variant_type != "baseline":
            print("   Calculating metrics vs baseline...")
            
            baseline = db.query(ModelVariant).filter(
                ModelVariant.experiment_id == experiment.id,
                ModelVariant.variant_type == "baseline"
            ).first()
            
            if baseline:
                baseline_outputs = db.query(GeneratedOutput).filter(
                    GeneratedOutput.variant_id == baseline.id,
                    GeneratedOutput.is_successful == 1
                ).all()
                
                if baseline_outputs:
                    baseline_texts = []
                    candidate_texts = []
                    
                    for output in outputs:
                        baseline_out = next(
                            (b for b in baseline_outputs if b.sample_id == output.sample_id),
                            None
                        )
                        if baseline_out:
                            baseline_texts.append(baseline_out.output_text)
                            candidate_texts.append(output.output_text)
                    
                    if baseline_texts:
                        bert_scores = bert_calc.calculate(candidate_texts, baseline_texts)
                        metrics.bertscore_precision_vs_baseline = bert_scores["precision"]
                        metrics.bertscore_recall_vs_baseline = bert_scores["recall"]
                        metrics.bertscore_f1_vs_baseline = bert_scores["f1"]
                        
                        metrics.cosine_similarity_vs_baseline = sim_calc.calculate_average_similarity(
                            candidate_texts, baseline_texts
                        )
                        
                        metrics.output_divergence_score = 1.0 - metrics.cosine_similarity_vs_baseline
        
        # Update status
        metrics.evaluation_status = "completed"
        metrics.samples_evaluated = len(outputs)
        db.commit()
        
        result = {
            "status": "completed",
            "variant_id": variant_id,
            "samples_evaluated": len(outputs),
            "metrics": {
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_tokens_per_second": metrics.avg_tokens_per_second,
                "bertscore_f1_vs_gt": metrics.bertscore_f1_vs_gt,
                "bertscore_f1_vs_baseline": metrics.bertscore_f1_vs_baseline,
                "cosine_similarity_vs_baseline": metrics.cosine_similarity_vs_baseline,
                "output_divergence": metrics.output_divergence_score
            }
        }
        
        print(f"✅ Evaluation complete for variant {variant_id}")
        return result
        
    except Exception as e:
        # Update metrics status on failure
        if metrics:
            metrics.evaluation_status = "failed"
            metrics.evaluation_error = str(e)
            db.commit()
        
        print(f"❌ Evaluation failed: {e}")
        raise
        
    finally:
        db.close()