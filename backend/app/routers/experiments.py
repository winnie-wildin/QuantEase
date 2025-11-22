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


@router.post("/{experiment_id}/samples")
async def upload_samples(
    experiment_id: int,
    samples: List[dict],
    db: Session = Depends(get_db)
):
    """Upload test samples for an experiment"""
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Check if samples have ground truth
    has_ground_truth = False
    ground_truth_count = 0
    
    created_samples = []
    for i, sample_data in enumerate(samples):
        input_text = sample_data.get('input_text')
        ground_truth = sample_data.get('ground_truth_output')
        
        if not input_text:
            raise HTTPException(
                status_code=400, 
                detail=f"Sample {i+1}: Missing 'input_text' field"
            )
        
        # Track ground truth
        if ground_truth and ground_truth.strip():
            ground_truth_count += 1
            has_ground_truth = True
        
        sample = DatasetSample(
            experiment_id=experiment_id,
            position=i,
            input_text=input_text,
            ground_truth_output=ground_truth if ground_truth else None
        )
        db.add(sample)
        created_samples.append(sample)
    
    # Update experiment with ground truth info
    exp.has_ground_truth = has_ground_truth
    exp.sample_count = len(created_samples)
    
    db.commit()
    
    print(f"✅ Uploaded {len(created_samples)} samples")
    print(f"📊 Ground truth: {ground_truth_count}/{len(created_samples)} samples")
    
    return {
        "message": f"Uploaded {len(created_samples)} samples",
        "sample_count": len(created_samples),
        "has_ground_truth": has_ground_truth,
        "ground_truth_count": ground_truth_count
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


# @router.post("/{experiment_id}/generate")
# async def trigger_generation(
#     experiment_id: int,
#     db: Session = Depends(get_db)
# ):
#     """Trigger generation for all variants in experiment"""
#     from app.tasks.baseline_generation import generate_baseline_outputs
#     from app.tasks.quantized_generation import generate_quantized_outputs
    
#     exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
#     if not exp:
#         raise HTTPException(status_code=404, detail="Experiment not found")
    
#     variants = db.query(ModelVariant).filter(
#         ModelVariant.experiment_id == experiment_id
#     ).all()
    
#     if not variants:
#         raise HTTPException(status_code=400, detail="No variants found")
    
#     # Trigger generation tasks
#     tasks = []
#     for variant in variants:
#         if variant.variant_type == "baseline":
#             task = generate_baseline_outputs.delay(experiment_id, variant.id)
#         else:
#             task = generate_quantized_outputs.delay(experiment_id, variant.id)
#         tasks.append({"variant_id": variant.id, "task_id": task.id})
    
#     exp.status = "generating"
#     db.commit()
    
#     return {
#         "message": "Generation started",
#         "experiment_id": experiment_id,
#         "tasks": tasks
#     }
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
        tasks.append({"variant_id": variant.id, "task_id": task.id})
    exp.is_draft = False 
    exp.status = "generating"
    db.commit()
    
    # Check completion
    import time
    time.sleep(2)
    
    for variant in variants:
        db.refresh(variant)
    
    all_completed = all(v.status == "completed" for v in variants)
    if all_completed:
        exp.status = "completed"
        db.commit()
    
    return {
        "message": "Generation completed" if all_completed else "Generation started",
        "experiment_id": experiment_id,
        "status": exp.status,
        "has_ground_truth": exp.has_ground_truth,
        "tasks": tasks
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
    db: Session = Depends(get_db)
):
    """Trigger evaluation for all variants"""
    from app.tasks.evaluation_task import evaluate_variant
    
    variants = db.query(ModelVariant).filter(
        ModelVariant.experiment_id == experiment_id,
        ModelVariant.status == "completed"
    ).all()
    
    if not variants:
        raise HTTPException(status_code=400, detail="No completed variants found")
    
    tasks = []
    for variant in variants:
        task = evaluate_variant.delay(variant.id)
        tasks.append({"variant_id": variant.id, "task_id": task.id})
    
    return {
        "message": "Evaluation started",
        "tasks": tasks
    }
# @router.get("/{experiment_id}/samples/comparison")
# async def get_sample_comparison(
#     experiment_id: int,
#     page: int = 1,
#     page_size: int = 20,
#     db: Session = Depends(get_db)
# ):
#     """Get side-by-side comparison of all samples with outputs"""
#     from app.utils.similarity_calculator import SimilarityCalculator
    
#     exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
#     if not exp:
#         raise HTTPException(status_code=404, detail="Experiment not found")
    
#     # Get all samples
#     samples = db.query(DatasetSample).filter(
#         DatasetSample.experiment_id == experiment_id
#     ).order_by(DatasetSample.position).all()
    
#     # Get variants (baseline optional)
#     baseline = db.query(ModelVariant).filter(
#         ModelVariant.experiment_id == experiment_id,
#         ModelVariant.variant_type == "baseline"
#     ).first()
    
#     quantized_variants = db.query(ModelVariant).filter(
#         ModelVariant.experiment_id == experiment_id,
#         ModelVariant.variant_type == "quantized"
#     ).all()
    
#     # Build comparison data
#     comparison_data = []
#     sim_calc = SimilarityCalculator()
    
#     for sample in samples:
#         sample_data = {
#             "sample_id": sample.id,
#             "position": sample.position,
#             "input_text": sample.input_text,
#             "ground_truth": sample.ground_truth_output,
#             "baseline_output": None,
#             "baseline_latency": None,
#             "quantized_outputs": []
#         }
        
#         # Get baseline output if exists
#         if baseline:
#             baseline_output = db.query(GeneratedOutput).filter(
#                 GeneratedOutput.sample_id == sample.id,
#                 GeneratedOutput.variant_id == baseline.id
#             ).first()
            
#             if baseline_output:
#                 sample_data["baseline_output"] = baseline_output.output_text
#                 sample_data["baseline_latency"] = baseline_output.latency_ms
        
#         # Get quantized outputs
#         for quant_variant in quantized_variants:
#             quant_output = db.query(GeneratedOutput).filter(
#                 GeneratedOutput.sample_id == sample.id,
#                 GeneratedOutput.variant_id == quant_variant.id
#             ).first()
            
#             if quant_output:
#                 # Calculate similarity (against ground truth or baseline)
#                 reference = sample.ground_truth_output if exp.has_ground_truth else sample_data["baseline_output"]
#                 similarity_score = None
                
#                 if reference and quant_output.output_text:
#                     try:
#                         similarity_score = sim_calc.calculate_single(
#                             quant_output.output_text,
#                             reference
#                         )
#                     except:
#                         similarity_score = None
                
#                 sample_data["quantized_outputs"].append({
#                     "variant_id": quant_variant.id,
#                     "model_name": quant_variant.model_name,
#                     "output_text": quant_output.output_text,
#                     "latency_ms": quant_output.latency_ms,
#                     "similarity_score": similarity_score
#                 })
        
#         comparison_data.append(sample_data)
    
#     # Sort by worst similarity
#     def get_worst_similarity(item):
#         scores = [q["similarity_score"] for q in item["quantized_outputs"] if q["similarity_score"] is not None]
#         return min(scores) if scores else 1.0
    
#     comparison_data.sort(key=get_worst_similarity)
    
#     # Paginate
#     total = len(comparison_data)
#     start = (page - 1) * page_size
#     end = start + page_size
#     paginated_data = comparison_data[start:end]
    
#     return {
#         "experiment_id": experiment_id,
#         "has_ground_truth": exp.has_ground_truth,
#         "has_baseline": baseline is not None,
#         "quantized_count": len(quantized_variants),
#         "total_samples": total,
#         "page": page,
#         "page_size": page_size,
#         "total_pages": (total + page_size - 1) // page_size,
#         "samples": paginated_data
#     }
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

