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

def evaluate_text_generation_task(outputs, experiment, variant, db, enable_llm_judge):
    """Evaluate text generation task"""
    from app.utils.task_evaluators import TextGenEvaluator
    from app.utils.llm_judge import LLMJudge
    import os
    
    print("   🔤 Evaluating Text Generation...")
    
    evaluator = TextGenEvaluator()
    
    # Prepare data
    candidate_texts = [o.output_text for o in outputs]
    
    # Get references (ground truth or baseline)
    if experiment.has_ground_truth:
        reference_texts = [o.sample.ground_truth_output for o in outputs]
        reference_type = "ground_truth"
    else:
        baseline = db.query(ModelVariant).filter(
            ModelVariant.experiment_id == experiment.id,
            ModelVariant.variant_type == "baseline"
        ).first()
        
        if not baseline:
            raise ValueError("No reference available (no ground truth or baseline)")
        
        baseline_outputs = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == baseline.id,
            GeneratedOutput.is_successful == 1
        ).all()
        
        reference_texts = [
            next((b.output_text for b in baseline_outputs if b.sample_id == o.sample_id), None)
            for o in outputs
        ]
        reference_type = "baseline"
    
    # Calculate metrics
    results = evaluator.evaluate(
        candidates=candidate_texts,
        references=reference_texts,
        baseline_outputs=reference_texts if reference_type == "baseline" else None
    )
    
    results['reference_type'] = reference_type
    
    # ✅ ENHANCED: LLM Judge with detailed logging
    print(f"   📊 enable_llm_judge parameter: {enable_llm_judge}")
    
    if enable_llm_judge:
        print(f"   🤖 LLM Judge requested - checking requirements...")
        
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            print(f"   ✅ GROQ_API_KEY found: {groq_key[:10]}...{groq_key[-4:]}")
            print(f"   🚀 Starting LLM Judge for {len(candidate_texts)} samples...")
            
            try:
                judge = LLMJudge()
                judge_percentage = experiment.judge_sample_percentage or 10.0
                
                print(f"   🎲 Sampling {judge_percentage}% of {len(candidate_texts)} samples...")
                
                judge_data = [
                    {
                        "input": outputs[i].sample.input_text,
                        "reference": reference_texts[i],
                        "candidate": candidate_texts[i]
                    }
                    for i in range(len(candidate_texts))
                ]
                
                sampled_data, indices = judge.sample_data(judge_data, judge_percentage)
                
                print(f"   📝 Judging {len(sampled_data)} samples: {indices}")
                
                judge_results = []
                for idx, item in enumerate(sampled_data):
                    print(f"   🔍 Judging sample {idx+1}/{len(sampled_data)} (original index: {indices[idx]})...")
                    
                    result = judge.judge_text_generation(
                        input_text=item["input"],
                        reference_output=item["reference"],
                        candidate_output=item["candidate"]
                    )
                    
                    if "error" in result:
                        print(f"      ❌ Judge error: {result['error']}")
                    else:
                        print(f"      ✅ Scores - Accuracy: {result.get('accuracy', 'N/A')}, Fluency: {result.get('fluency', 'N/A')}, Coherence: {result.get('coherence', 'N/A')}")
                    
                    judge_results.append(result)
                
                print(f"   🧮 Aggregating {len(judge_results)} judge results...")
                aggregated = judge.aggregate_judge_scores(judge_results, "text_gen")
                
                results["llm_judge"] = {
                    **aggregated,
                    "sample_percentage": judge_percentage,
                    "sampled_indices": indices
                }
                
                print(f"   ✅ LLM Judge complete!")
                print(f"      📊 Avg Accuracy: {aggregated.get('avg_accuracy', 'N/A')}")
                print(f"      📊 Avg Fluency: {aggregated.get('avg_fluency', 'N/A')}")
                print(f"      📊 Avg Coherence: {aggregated.get('avg_coherence', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ LLM Judge FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                results["llm_judge"] = {"error": str(e)}
        else:
            print(f"   ❌ GROQ_API_KEY not found - skipping LLM judge")
            results["llm_judge"] = {"error": "GROQ_API_KEY not configured"}
    else:
        print(f"   ℹ️  LLM Judge disabled (enable_llm_judge=False)")
    
    return results



def evaluate_classification_task(outputs, experiment, variant, db):
    """Evaluate classification task"""
    from app.utils.task_evaluators import ClassificationEvaluator
    
    print("   🎯 Evaluating Classification...")
    
    evaluator = ClassificationEvaluator()
    
    # Get predictions and true labels
    predictions = [o.output_text.strip() for o in outputs]
    
    if not experiment.has_ground_truth:
        raise ValueError("Classification requires ground truth labels")
    
    true_labels = [o.sample.ground_truth_output.strip() for o in outputs]
    
    # Evaluate
    results = evaluator.evaluate(
        predictions=predictions,
        true_labels=true_labels
    )
    
    return results


def evaluate_rag_task(outputs, experiment, variant, db, enable_llm_judge):
    """Evaluate RAG task"""
    from app.utils.task_evaluators import RAGEvaluator
    from app.utils.llm_judge import LLMJudge
    import os
    
    print("   🔍 Evaluating RAG...")
    
    evaluator = RAGEvaluator()
    
    # Get answers
    generated_answers = [o.output_text for o in outputs]
    
    # Get references
    if experiment.has_ground_truth:
        reference_answers = [o.sample.ground_truth_output for o in outputs]
        reference_type = "ground_truth"
    else:
        baseline = db.query(ModelVariant).filter(
            ModelVariant.experiment_id == experiment.id,
            ModelVariant.variant_type == "baseline"
        ).first()
        
        if not baseline:
            raise ValueError("No reference available for RAG")
        
        baseline_outputs = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == baseline.id,
            GeneratedOutput.is_successful == 1
        ).all()
        
        reference_answers = [
            next((b.output_text for b in baseline_outputs if b.sample_id == o.sample_id), None)
            for o in outputs
        ]
        reference_type = "baseline"
    
    # Calculate metrics
    results = evaluator.evaluate(
        generated_answers=generated_answers,
        reference_answers=reference_answers
    )
    
    results['reference_type'] = reference_type
    
    # ✅ ENHANCED: LLM Judge with detailed logging
    print(f"   📊 enable_llm_judge parameter: {enable_llm_judge}")
    
    if enable_llm_judge:
        print(f"   🤖 LLM Judge requested for hallucination detection...")
        
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            print(f"   ✅ GROQ_API_KEY found")
            print(f"   🚀 Starting hallucination check for {len(generated_answers)} samples...")
            
            try:
                judge = LLMJudge()
                judge_percentage = experiment.judge_sample_percentage or 10.0
                
                print(f"   🎲 Sampling {judge_percentage}% of {len(generated_answers)} samples...")
                
                judge_data = [
                    {
                        "question": outputs[i].sample.input_text,
                        "context": outputs[i].sample.context or "",
                        "reference": reference_answers[i],
                        "candidate": generated_answers[i]
                    }
                    for i in range(len(generated_answers))
                ]
                
                sampled_data, indices = judge.sample_data(judge_data, judge_percentage)
                
                print(f"   📝 Checking {len(sampled_data)} samples for hallucinations: {indices}")
                
                judge_results = []
                for idx, item in enumerate(sampled_data):
                    print(f"   🔍 Checking sample {idx+1}/{len(sampled_data)} (original index: {indices[idx]})...")
                    
                    result = judge.judge_rag_factuality(
                        question=item["question"],
                        context=item["context"],
                        reference_answer=item["reference"],
                        candidate_answer=item["candidate"]
                    )
                    
                    if "error" in result:
                        print(f"      ❌ Judge error: {result['error']}")
                    else:
                        halluc = "YES ⚠️" if result.get('has_hallucination') else "NO ✅"
                        print(f"      Hallucination: {halluc}, Factual: {result.get('factual_correctness', 'N/A')}/5, Complete: {result.get('completeness', 'N/A')}/5")
                    
                    judge_results.append(result)
                
                print(f"   🧮 Aggregating {len(judge_results)} judge results...")
                aggregated = judge.aggregate_judge_scores(judge_results, "rag")
                
                results["llm_judge"] = {
                    **aggregated,
                    "sample_percentage": judge_percentage,
                    "sampled_indices": indices
                }
                
                print(f"   ✅ Hallucination check complete!")
                print(f"      📊 Hallucination Rate: {aggregated.get('hallucination_rate', 'N/A')}%")
                print(f"      📊 Avg Factual Correctness: {aggregated.get('avg_factual_correctness', 'N/A')}/5")
                
            except Exception as e:
                print(f"   ❌ Hallucination check FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                results["llm_judge"] = {"error": str(e)}
        else:
            print(f"   ❌ GROQ_API_KEY not found - skipping hallucination check")
            results["llm_judge"] = {"error": "GROQ_API_KEY not configured"}
    else:
        print(f"   ℹ️  LLM Judge disabled (enable_llm_judge=False)")
    
    return results


def run_legacy_evaluation(outputs, experiment, variant, db):
    """Legacy evaluation for backward compatibility with old experiments"""
    from app.utils.bertscore_calculator import BERTScoreCalculator
    from app.utils.similarity_calculator import SimilarityCalculator
    
    print("   📊 Running legacy evaluation...")
    
    bert_calc = BERTScoreCalculator()
    sim_calc = SimilarityCalculator()
    
    candidate_texts = [o.output_text for o in outputs]
    
    # Try to get references
    if experiment.has_ground_truth:
        reference_texts = [o.sample.ground_truth_output for o in outputs]
        
        bert_scores = bert_calc.calculate(candidate_texts, reference_texts)
        cosine_sim = sim_calc.calculate_average_similarity(candidate_texts, reference_texts)
        
        return {
            "bertscore_f1_vs_gt": bert_scores["f1"],
            "cosine_similarity_vs_gt": cosine_sim,
            "evaluation_type": "legacy_with_gt"
        }
    
    return {
        "evaluation_type": "legacy_no_reference",
        "message": "No evaluation possible without reference"
    }

@celery_app.task(bind=True, name="evaluate_variant")
def evaluate_variant(
    self, 
    variant_id: int,
    enable_llm_judge: bool = False
):
    """
    Task-aware evaluation: routes to appropriate evaluator based on task type.
    
    Args:
        variant_id: ID of the variant to evaluate
        enable_llm_judge: Whether to use LLM judge (optional, costs money)
    
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
        task_type = experiment.task_type or 'text_generation'  # Default for old experiments
        
        print(f"📊 Evaluating variant {variant_id} for {task_type} task...")
        print(f"   🤖 LLM Judge: {'ENABLED ✅' if enable_llm_judge else 'DISABLED ❌'}")
        if enable_llm_judge:
            import os
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                print(f"   🔑 GROQ_API_KEY found: {groq_key[:10]}...{groq_key[-4:]}")
            else:
                print(f"   ⚠️  GROQ_API_KEY NOT FOUND - Judge will fail!")
        
        # Get all outputs
        outputs = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == variant_id,
            GeneratedOutput.is_successful == 1
        ).all()
        
        if not outputs:
            raise ValueError(f"No successful outputs found for variant {variant_id}")
        
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
        
        # ===== CALCULATE OLD METRICS (deprecated but kept) =====
        latencies = [o.latency_ms for o in outputs if o.latency_ms]
        token_counts = [o.token_count for o in outputs if o.token_count]
        speeds = [o.tokens_per_second for o in outputs if o.tokens_per_second]
        
        metrics.avg_latency_ms = float(np.mean(latencies)) if latencies else None
        metrics.avg_token_count = float(np.mean(token_counts)) if token_counts else None
        metrics.avg_tokens_per_second = float(np.mean(speeds)) if speeds else None
        
        if variant.model_path:
            import os
            if os.path.exists(variant.model_path):
                metrics.model_size_mb = os.path.getsize(variant.model_path) / (1024 * 1024)
        
        # ===== TASK-AWARE EVALUATION =====
        evaluation_results = None
        
        if task_type == 'text_generation':
            evaluation_results = evaluate_text_generation_task(
                outputs, experiment, variant, db, enable_llm_judge
            )
        
        elif task_type == 'classification':
            evaluation_results = evaluate_classification_task(
                outputs, experiment, variant, db
            )
        
        elif task_type == 'rag':
            evaluation_results = evaluate_rag_task(
                outputs, experiment, variant, db, enable_llm_judge
            )
        
        else:
            # Fallback to old evaluation for backward compatibility
            print(f"⚠️ Unknown task type '{task_type}', using legacy evaluation")
            evaluation_results = run_legacy_evaluation(
                outputs, experiment, variant, db
            )
        
        # Store results
        metrics.evaluation_results = evaluation_results
        metrics.evaluation_status = "completed"
        metrics.samples_evaluated = len(outputs)
        db.commit()
        
        print(f"✅ Evaluation complete for variant {variant_id}")
        return {
            "status": "completed",
            "variant_id": variant_id,
            "task_type": task_type,
            "samples_evaluated": len(outputs),
            "results": evaluation_results
        }
        
    except Exception as e:
        if metrics:
            metrics.evaluation_status = "failed"
            metrics.evaluation_error = str(e)
            db.commit()
        
        print(f"❌ Evaluation failed: {e}")
        raise
        
    finally:
        db.close()