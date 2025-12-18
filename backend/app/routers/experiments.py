#app/routers/experiments.py
"""
Experiments API Router
Handles experiment CRUD and generation triggers
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import (
    Experiment, DatasetSample, ModelVariant, 
    ComparativeMetrics, GeneratedOutput
)
from pydantic import BaseModel
from datetime import datetime
import json
from app.utils.json_normalizer import JSONNormalizer
router = APIRouter(prefix="/experiments", tags=["experiments"])


# ===== PYDANTIC SCHEMAS =====

class ExperimentCreate(BaseModel):
    name: str
    baseline_model_id: int
    has_ground_truth: bool = False
    generate_baseline: bool = True


class ExperimentResponse(BaseModel):
    id: int
    name: str
    baseline_model_id: int
    has_ground_truth: bool
    sample_count: int
    status: str
    progress: int
    created_at: datetime
    updated_at: datetime | None = None
    task_type: str | None = None


    class Config:
        from_attributes = True


class SampleUpload(BaseModel):
    input_text: str
    ground_truth_output: str | None = None


class VariantCreate(BaseModel):
    variant_type: str
    model_name: str
    quantization_level: str | None = None
    model_path: str | None = None
    inference_provider: str


class VariantResponse(BaseModel):
    id: int
    experiment_id: int
    variant_type: str
    model_name: str
    quantization_level: str | None
    inference_provider: str
    status: str
    progress: float
    display_name: str

    class Config:
        from_attributes = True


class MetricsResponse(BaseModel):
    id: int
    variant_id: int
    model_size_mb: float | None
    avg_latency_ms: float | None
    avg_token_count: float | None
    avg_tokens_per_second: float | None
    bertscore_f1_vs_gt: float | None
    bertscore_f1_vs_baseline: float | None
    cosine_similarity_vs_gt: float | None
    cosine_similarity_vs_baseline: float | None
    output_divergence_score: float | None
    samples_evaluated: int
    evaluation_status: str
    evaluation_results: dict | None = None

    class Config:
        from_attributes = True


# ===== API ENDPOINTS =====

@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    include_drafts: bool = False,
    db: Session = Depends(get_db)
):
    """Get all experiments (excluding drafts by default)"""
    query = db.query(Experiment)
    
    if not include_drafts:
        query = query.filter(Experiment.is_draft == False)
    
    experiments = query.order_by(Experiment.created_at.desc()).all()
    return experiments


@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    exp: ExperimentCreate,
    db: Session = Depends(get_db)
):
    """Create a new experiment"""
    new_exp = Experiment(
        name=exp.name,
        baseline_model_id=exp.baseline_model_id,
        has_ground_truth=exp.has_ground_truth,
        generate_baseline=exp.generate_baseline,
        status="created",
        sample_count=0,
        progress=0
    )
    db.add(new_exp)
    db.commit()
    db.refresh(new_exp)
    return new_exp


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Get experiment by ID"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp
from pydantic import BaseModel

# Add this Pydantic model at the top (after other imports)
class SamplesUploadRequest(BaseModel):
    samples: List[dict]
    task_type: str

# Then UPDATE the endpoint signature (around line 143):
@router.post("/{experiment_id}/samples")
async def upload_samples(
    experiment_id: int,
    data: SamplesUploadRequest,  # ✅ Accept JSON body as model
    db: Session = Depends(get_db)
):
    print(f"📥 Received request: task_type={data.task_type}, samples count={len(data.samples)}")
    
    """Upload test samples for an experiment with task-aware normalization"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Access data from the model
    task_type = data.task_type
    samples = data.samples
    
    # Validate task type
    valid_task_types = ['text_generation', 'classification', 'rag']
    if task_type not in valid_task_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type. Must be one of: {valid_task_types}"
        )
    # ✅ NEW: Validate data format matches task type
    if len(samples) > 0:
        first_sample = samples[0]
        has_context = 'context' in first_sample
        has_label_style_output = 'label' in first_sample or (
            'ground_truth_output' in first_sample and 
            len(str(first_sample.get('ground_truth_output', '')).split()) <= 3
        )
        
        # Check for mismatches
        if task_type == 'rag' and not has_context:
            raise HTTPException(
                status_code=400,
                detail="❌ RAG task requires 'context' field in dataset. Your data appears to be text generation format."
            )
        
        if task_type == 'text_generation' and has_context:
            raise HTTPException(
                status_code=400,
                detail="❌ You selected 'Text Generation' but your dataset has 'context' field. This is RAG format! Please select 'RAG' task type instead."
            )
        
        if task_type == 'classification' and has_context:
            raise HTTPException(
                status_code=400,
                detail="❌ Classification task should not have 'context' field. Your data appears to be RAG format."
            )
    
    # Normalize JSON based on task type
    try:
        normalized_data, metadata = JSONNormalizer.normalize_by_task(samples, task_type)
        print(f"✅ Normalization successful: {metadata}")
    except Exception as e:
        print(f"❌ Normalization failed: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        raise HTTPException(
            status_code=400,
            detail=f"JSON normalization failed: {str(e)}"
        )
    
    # Create dataset samples
    created_samples = []
    for i, sample_data in enumerate(normalized_data):
        sample = DatasetSample(
            experiment_id=experiment_id,
            position=i,
            input_text=sample_data['input'],
            ground_truth_output=sample_data.get('output') or sample_data.get('label'),
            context=sample_data.get('context')  # Only for RAG tasks
        )
        db.add(sample)
        created_samples.append(sample)
    
    # Updated experiment with task-aware metadata
    exp.task_type = task_type
    exp.normalization_metadata = metadata
    exp.has_ground_truth = metadata['has_ground_truth']
    exp.sample_count = len(created_samples)
    
    # Extract and store labels for classification tasks
    if task_type == 'classification':
        from app.utils.task_prompt_builder import TaskPromptBuilder
        try:
            labels = TaskPromptBuilder.extract_labels_from_samples(samples)
            exp.detected_labels = labels  # Store as list in JSON field
            print(f"🏷️  Detected {len(labels)} labels: {labels}")
        except Exception as e:
            print(f"⚠️  Could not extract labels: {e}")
            exp.detected_labels = []
    else:
        exp.detected_labels = None
    
    db.commit()
    
    print(f"✅ Uploaded {len(created_samples)} samples for {task_type} task")
    print(f"📊 Has ground truth: {metadata['has_ground_truth']}")
    if task_type == 'classification' and 'class_statistics' in metadata:
        print(f"📊 Classes detected: {metadata['class_statistics']['num_classes']}")
    
    return {
        "message": f"Uploaded {len(created_samples)} samples",
        "task_type": task_type,
        "sample_count": len(created_samples),
        "has_ground_truth": metadata['has_ground_truth'],
        "metadata": metadata
    }
@router.post("/{experiment_id}/variants", response_model=VariantResponse)
async def create_variant(
    experiment_id: int,
    variant: VariantCreate,
    db: Session = Depends(get_db)
):
    """Create a model variant for comparison"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    new_variant = ModelVariant(
        experiment_id=experiment_id,
        variant_type=variant.variant_type,
        model_name=variant.model_name,
        quantization_level=variant.quantization_level,
        model_path=variant.model_path,
        inference_provider=variant.inference_provider,
        status="pending",
        progress=0.0
    )
    db.add(new_variant)
    db.commit()
    db.refresh(new_variant)
    return new_variant


@router.post("/{experiment_id}/generate")
async def trigger_generation(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Trigger generation for all variants in experiment"""
    from app.tasks.baseline_generation import generate_baseline_outputs
    from app.tasks.quantized_generation import generate_quantized_outputs
    
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id
    ).all()
    
    if not variants:
        raise HTTPException(status_code=400, detail="No variants found")
    
    # Validate: If no ground truth, must have baseline
    has_baseline = any(v.variant_type == "baseline" for v in variants)
    if not exp.has_ground_truth and not has_baseline:
        raise HTTPException(
            status_code=400, 
            detail="Baseline model required when no ground truth is provided"
        )
    
    # Trigger generation tasks only for variants that need generation
    tasks = []
    for variant in variants:
        if variant.variant_type == "baseline":
            task = generate_baseline_outputs.delay(experiment_id, variant.id)
        else:
            task = generate_quantized_outputs.delay(experiment_id, variant.id)
        # Store task ID in variant for cancellation support
        variant.celery_task_id = task.id
        tasks.append({"variant_id": variant.id, "task_id": task.id})
    exp.is_draft = False 
    exp.status = "generating"
    db.commit()
    
    # Return immediately - don't wait for completion
    # The frontend will poll for status updates
    return {
        "message": "Generation started",
        "experiment_id": experiment_id,
        "status": exp.status,
        "has_ground_truth": exp.has_ground_truth,
        "tasks": tasks
    }


@router.get("/{experiment_id}/generation-status")
async def get_generation_status(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Get real-time generation progress for all variants"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id
    ).all()
    
    # Get total samples
    total_samples = db.query(DatasetSample).filter(
        DatasetSample.experiment_id == experiment_id
    ).count()
    
    variant_statuses = []
    for variant in variants:
        # Count completed outputs (this is the source of truth - always fresh from DB)
        # This query always hits the database, so it's always up-to-date
        completed = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == variant.id
        ).count()
        
        # Calculate progress from actual completed count (more reliable than variant.progress)
        # This is the real-time progress based on actual database records
        actual_progress = completed / total_samples if total_samples > 0 else 0.0
        
        # Refresh variant to get latest status
        db.refresh(variant)
        
        # Update variant.progress if it's out of sync (helps keep it accurate for other queries)
        if abs(variant.progress - actual_progress) > 0.01:  # Only update if significantly different
            variant.progress = actual_progress
            db.commit()
        
        # Get last 3 completed samples for live preview
        recent_outputs = db.query(GeneratedOutput).filter(
            GeneratedOutput.variant_id == variant.id
        ).order_by(GeneratedOutput.id.desc()).limit(3).all()
        
        recent_samples = []
        for output in recent_outputs:
            sample = db.query(DatasetSample).filter(
                DatasetSample.id == output.sample_id
            ).first()
            if sample:
                recent_samples.append({
                    "position": sample.position,
                    "output_preview": output.output_text[:80] + "..." if len(output.output_text) > 80 else output.output_text,
                    "latency_ms": output.latency_ms,
                    "is_successful": bool(output.is_successful),
                    "timestamp": output.id  # Use ID as proxy for order
                })
        
        variant_statuses.append({
            "variant_id": variant.id,
            "model_name": variant.model_name,
            "variant_type": variant.variant_type,
            "status": variant.status,
            "progress": actual_progress,  # Use calculated progress instead of variant.progress
            "completed_samples": completed,
            "total_samples": total_samples,
            "recent_samples": recent_samples
        })
    
    # Check if all variants are completed, cancelled, or failed
    all_finished = all(v["status"] in ("completed", "cancelled", "failed") for v in variant_statuses)
    
    # Also update experiment status if all variants are finished (fixes stuck experiments)
    if all_finished and exp.status == "generating":
        # Determine experiment status based on variant statuses
        has_cancelled = any(v["status"] == "cancelled" for v in variant_statuses)
        has_failed = any(v["status"] == "failed" for v in variant_statuses)
        has_completed = any(v["status"] == "completed" for v in variant_statuses)
        
        if has_cancelled:
            exp.status = "cancelled"
        elif has_failed and not has_completed:
            exp.status = "failed"
        else:
            exp.status = "completed"
        db.commit()
    
    return {
        "experiment_id": experiment_id,
        "experiment_status": exp.status,
        "total_samples": total_samples,
        "variants": variant_statuses,
        "all_completed": all_finished
    }

@router.get("/{experiment_id}/variants", response_model=List[VariantResponse])
async def get_variants(experiment_id: int, db: Session = Depends(get_db)):
    """Get all variants for an experiment"""
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id
    ).all()
    return variants


@router.get("/{experiment_id}/metrics", response_model=List[MetricsResponse])
async def get_metrics(experiment_id: int, db: Session = Depends(get_db)):
    """Get comparison metrics for all variants"""
    metrics = db.query(ComparativeMetrics).filter(
        ComparativeMetrics.experiment_id == experiment_id
    ).all()
    return metrics

@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Delete an experiment and all related data"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Delete cascade should handle related records
    db.delete(exp)
    db.commit()
    
    return {"message": f"Experiment {experiment_id} deleted"}

@router.post("/{experiment_id}/evaluate")
async def trigger_evaluation(
    experiment_id: int,
    enable_llm_judge: bool = False,  # NEW: Optional LLM judge
    db: Session = Depends(get_db)
):
    """Trigger evaluation for all variants"""
    from app.tasks.evaluation_task import evaluate_variant
    
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id,
        ModelVariant.status == "completed"
    ).all()
    
    if not variants:
        raise HTTPException(status_code=400, detail="No completed variants found")
    
    # Pass enable_llm_judge to evaluation task
    tasks = []
    for variant in variants:
        task = evaluate_variant.delay(
            variant.id,
            enable_llm_judge=enable_llm_judge or exp.judge_enabled  # Use experiment setting if not specified
        )
        tasks.append({"variant_id": variant.id, "task_id": task.id})
    
    return {
        "message": "Evaluation started",
        "experiment_id": experiment_id,
        "llm_judge_enabled": enable_llm_judge or exp.judge_enabled,
        "tasks": tasks
    }

@router.post("/{experiment_id}/cancel-generation")
async def cancel_generation(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Cancel running generation tasks for an experiment"""
    from app.tasks.celery_app import celery_app
    
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get all variants that might need cancellation (generating, pending, or stuck)
    # This handles cases where status might be stuck or inconsistent
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id,
        ModelVariant.status.in_(["generating", "pending"])
    ).all()
    
    if not variants:
        # Check if there are any variants at all - might already be completed/cancelled
        all_variants = db.query(ModelVariant).filter(
            ModelVariant.experiment_id == experiment_id
        ).all()
        
        if not all_variants:
            raise HTTPException(
                status_code=404,
                detail="No variants found for this experiment"
            )
        else:
            # Variants exist but none are in generating/pending state
            raise HTTPException(
                status_code=400,
                detail="No active generation tasks to cancel. All variants are already completed, cancelled, or failed."
            )
    
    cancelled_tasks = []
    for variant in variants:
        try:
            # First set status to cancelled (so task can check and exit gracefully)
            variant.status = "cancelled"
            variant.error_message = "Generation cancelled by user"
            
            # Try to revoke the task if we have a task ID
            if variant.celery_task_id:
                try:
                    # Revoke the task (terminate=True will kill running task)
                    celery_app.control.revoke(variant.celery_task_id, terminate=True)
                    variant.celery_task_id = None
                except Exception as e:
                    print(f"Error revoking task {variant.celery_task_id}: {e}")
                    # Continue anyway - status is already set to cancelled
            else:
                # No task ID (old stuck task) - just mark as cancelled
                print(f"Variant {variant.id} has no celery_task_id, marking as cancelled anyway")
            
            cancelled_tasks.append(variant.id)
        except Exception as e:
            print(f"Error cancelling variant {variant.id}: {e}")
            # Still mark as cancelled even if there's an error
            variant.status = "cancelled"
            variant.error_message = f"Generation cancelled (error: {str(e)})"
            cancelled_tasks.append(variant.id)
    
    # Update experiment status
    if len(cancelled_tasks) > 0:
        exp.status = "cancelled"
        db.commit()
        
        return {
            "message": f"Cancelled generation for {len(cancelled_tasks)} variant(s)",
            "experiment_id": experiment_id,
            "cancelled_variants": cancelled_tasks
        }
    else:
        # No variants were cancelled, but let's check if experiment status needs updating
        all_variants = db.query(ModelVariant).filter(
            ModelVariant.experiment_id == experiment_id
        ).all()
        
        # If all variants are completed/failed/cancelled, update experiment status
        if all(v.status in ("completed", "failed", "cancelled") for v in all_variants):
            if any(v.status == "failed" for v in all_variants) and not any(v.status == "completed" for v in all_variants):
                exp.status = "failed"
            elif any(v.status == "cancelled" for v in all_variants):
                exp.status = "cancelled"
            else:
                exp.status = "completed"
            db.commit()
        
        return {
            "message": "No active generation tasks found. Experiment status updated.",
            "experiment_id": experiment_id,
            "experiment_status": exp.status,
            "cancelled_variants": []
        }

@router.get("/{experiment_id}/samples/comparison")
async def get_sample_comparison(
    experiment_id: int,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    """Get side-by-side comparison of all samples with outputs"""
    from app.utils.similarity_calculator import SimilarityCalculator
    
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Get all samples
    samples = db.query(DatasetSample).filter(
        DatasetSample.experiment_id == experiment_id
    ).order_by(DatasetSample.position).all()
    
    # Get variants
    baseline = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id,
        ModelVariant.variant_type == "baseline"
    ).first()
    
    quantized_variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id,
        ModelVariant.variant_type == "quantized"
    ).all()
    
    # Build comparison data
    comparison_data = []
    sim_calc = SimilarityCalculator()
    
    for sample in samples:
        sample_data = {
            "sample_id": sample.id,
            "position": sample.position,
            "input_text": sample.input_text,
            "ground_truth": sample.ground_truth_output,  # ← Include this
            "baseline_output": None,
            "baseline_latency": None,
            "quantized_outputs": []
        }
        
        # Get baseline output if exists
        if baseline:
            baseline_output = db.query(GeneratedOutput).filter(
                GeneratedOutput.sample_id == sample.id,
                GeneratedOutput.variant_id == baseline.id
            ).first()
            
            if baseline_output:
                sample_data["baseline_output"] = baseline_output.output_text
                sample_data["baseline_latency"] = baseline_output.latency_ms
        
        # Get quantized outputs
        for quant_variant in quantized_variants:
            quant_output = db.query(GeneratedOutput).filter(
                GeneratedOutput.sample_id == sample.id,
                GeneratedOutput.variant_id == quant_variant.id
            ).first()
            
            if quant_output:
                # Calculate similarity (against ground truth OR baseline)
                reference = sample.ground_truth_output if exp.has_ground_truth else sample_data["baseline_output"]
                similarity_score = None
                
                if reference and quant_output.output_text:
                    try:
                        similarity_score = sim_calc.calculate_single(
                            quant_output.output_text,
                            reference
                        )
                    except:
                        similarity_score = None
                
                sample_data["quantized_outputs"].append({
                    "variant_id": quant_variant.id,
                    "model_name": quant_variant.model_name,
                    "output_text": quant_output.output_text,
                    "latency_ms": quant_output.latency_ms,
                    "similarity_score": similarity_score
                })
        
        comparison_data.append(sample_data)
    
    # Sort by worst similarity if we have similarity scores
    def get_worst_similarity(item):
        scores = [q["similarity_score"] for q in item["quantized_outputs"] if q["similarity_score"] is not None]
        return min(scores) if scores else 1.0
    
    comparison_data.sort(key=get_worst_similarity)
    
    # Paginate
    total = len(comparison_data)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_data = comparison_data[start:end]
    
    return {
        "experiment_id": experiment_id,
        "has_ground_truth": exp.has_ground_truth,
        "has_baseline": baseline is not None,
        "quantized_count": len(quantized_variants),
        "total_samples": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "samples": paginated_data
    }

@router.get("/models/baseline")
async def list_baseline_models():
    """Get available baseline models"""
    from app.config.models import GROQ_MODELS
    return {"models": GROQ_MODELS}


@router.get("/models/quantized")
async def list_quantized_models():
    """Get available quantized models"""
    from app.config.models import QUANTIZED_MODELS
    return {"models": QUANTIZED_MODELS}

@router.get("/models/recommendations/{task_type}")
async def get_model_recommendations(task_type: str):
    """Get recommended baseline model for task type"""
    
    recommendations = {
        "text_generation": {
            "model_name": "llama-3.3-70b",
            "display_name": "Llama 3.3 70B",
            "reason": "Best for reasoning and long-form content generation",
            "provider": "groq"
        },
        "classification": {
            "model_name": "llama-3.1-8b",
            "display_name": "Llama 3.1 8B", 
            "reason": "Efficient and accurate for classification tasks",
            "provider": "groq"
        },
        "rag": {
            "model_name": "llama-3.3-70b",
            "display_name": "Llama 3.3 70B",
            "reason": "Excellent at understanding context and extracting answers",
            "provider": "groq"
        }
    }
    
    if task_type not in recommendations:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task type. Must be one of: {list(recommendations.keys())}"
        )
    
    return recommendations[task_type]