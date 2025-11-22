# from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
# from sqlalchemy.orm import Session
# from typing import List
# import json
# import csv
# import io

# from app.database import get_db
# from app.models import Experiment, DatasetSample, Model
# from app.schemas import ExperimentCreate, ExperimentResponse, ExperimentDetailResponse

# router = APIRouter()
# print("✅ DEBUG: Experiments router created")


# @router.post("/create", response_model=ExperimentResponse, status_code=201)
# async def create_experiment(
#     name: str = Form(...),
#     model_id: int = Form(...),
#     dataset_file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     """
#     Create a new experiment with dataset upload.
    
#     Args:
#         name: Experiment name
#         model_id: ID of the model to optimize
#         dataset_file: JSON or CSV file with dataset samples
        
#     Returns:
#         Created experiment details
        
#     Raises:
#         400: Invalid model_id or dataset format
#         404: Model not found
#     """
#     # Validate model exists
#     model = db.query(Model).filter(Model.id == model_id).first()
#     if not model:
#         raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
#     # Read and parse dataset file
#     content = await dataset_file.read()
#     content_str = content.decode('utf-8')
    
#     samples_data = []
#     has_outputs = False
    
#     try:
#         # Try parsing as JSON
#         if dataset_file.filename.endswith('.json'):
#             data = json.loads(content_str)
#             if 'samples' in data:
#                 samples_data = data['samples']
#             elif isinstance(data, list):
#                 samples_data = data
#             else:
#                 raise HTTPException(
#                     status_code=400, 
#                     detail="JSON must contain 'samples' array or be an array of objects"
#                 )
        
#         # Try parsing as CSV
#         elif dataset_file.filename.endswith('.csv'):
#             csv_reader = csv.DictReader(io.StringIO(content_str))
#             samples_data = list(csv_reader)
        
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="File must be .json or .csv"
#             )
        
#         if not samples_data:
#             raise HTTPException(status_code=400, detail="Dataset is empty")
        
#         # Check if dataset has expected outputs
#         first_sample = samples_data[0]
#         has_outputs = 'expected_output' in first_sample and first_sample['expected_output']
        
#         # Validate required fields
#         for idx, sample in enumerate(samples_data):
#             if 'input' not in sample or not sample['input']:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Sample at index {idx} missing required 'input' field"
#                 )
        
#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error parsing dataset: {str(e)}")
    
#     # Create experiment
#     experiment = Experiment(
#         name=name,
#         model_id=model_id,
#         status="pending",
#         has_reference_outputs=has_outputs,
#         sample_count=len(samples_data)
#     )
#     db.add(experiment)
#     db.flush()  # Get experiment ID
    
#     # Create dataset samples
#     for sample_data in samples_data:
#         dataset_sample = DatasetSample(
#             experiment_id=experiment.id,
#             input=sample_data['input'],
#             expected_output=sample_data.get('expected_output'),
#             is_selected=False
#         )
#         db.add(dataset_sample)
    
#     db.commit()
#     db.refresh(experiment)
    
#     return experiment


# @router.get("/", response_model=List[ExperimentResponse])
# def list_experiments(db: Session = Depends(get_db)):
#     """
#     List all experiments.
    
#     Returns:
#         List of all experiments
#     """
#     experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
#     return experiments


# @router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
# def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
#     """
#     Get experiment details including dataset samples.
    
#     Args:
#         experiment_id: ID of the experiment
        
#     Returns:
#         Experiment details with samples
        
#     Raises:
#         404: Experiment not found
#     """
#     experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
#     if not experiment:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Experiment with ID {experiment_id} not found"
#         )
#     return experiment


# @router.delete("/{experiment_id}", status_code=204)
# def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
#     """
#     Delete an experiment and all its samples.
    
#     Args:
#         experiment_id: ID of the experiment to delete
        
#     Raises:
#         404: Experiment not found
#     """
#     experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
#     if not experiment:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Experiment with ID {experiment_id} not found"
#         )
    
#     db.delete(experiment)
#     db.commit()
#     return None
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session, joinedload
from typing import List
import json
import csv
import io

from app.database import get_db
from app.models import Experiment, DatasetSample, Model
from app.schemas import ExperimentCreate, ExperimentResponse, ExperimentDetailResponse

router = APIRouter()
print("✅ DEBUG: Experiments router created")


@router.post("/create", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    name: str = Form(...),
    model_id: int = Form(...),
    dataset_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Create a new experiment with dataset upload.
    
    Args:
        name: Experiment name
        model_id: ID of the model to optimize
        dataset_file: JSON or CSV file with dataset samples
        
    Returns:
        Created experiment details
        
    Raises:
        400: Invalid model_id or dataset format
        404: Model not found
    """
    # Validate model exists
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Read and parse dataset file
    content = await dataset_file.read()
    content_str = content.decode('utf-8')
    
    samples_data = []
    has_outputs = False
    
    try:
        # Try parsing as JSON
        if dataset_file.filename.endswith('.json'):
            data = json.loads(content_str)
            if 'samples' in data:
                samples_data = data['samples']
            elif isinstance(data, list):
                samples_data = data
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="JSON must contain 'samples' array or be an array of objects"
                )
        
        # Try parsing as CSV
        elif dataset_file.filename.endswith('.csv'):
            csv_reader = csv.DictReader(io.StringIO(content_str))
            samples_data = list(csv_reader)
        
        else:
            raise HTTPException(
                status_code=400,
                detail="File must be .json or .csv"
            )
        
        if not samples_data:
            raise HTTPException(status_code=400, detail="Dataset is empty")
        
        # Check if dataset has expected outputs
        # FIX: Properly convert to boolean instead of returning the string value
        first_sample = samples_data[0]
        has_outputs = bool(first_sample.get('expected_output'))
        
        # Validate required fields
        for idx, sample in enumerate(samples_data):
            if 'input' not in sample or not sample['input']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample at index {idx} missing required 'input' field"
                )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing dataset: {str(e)}")
    
    # Create experiment
    experiment = Experiment(
        name=name,
        model_id=model_id,
        status="pending",
        has_reference_outputs=has_outputs,
        sample_count=len(samples_data)
    )
    db.add(experiment)
    db.flush()  # Get experiment ID
    
    # Create dataset samples
    for sample_data in samples_data:
        dataset_sample = DatasetSample(
            experiment_id=experiment.id,
            input=sample_data['input'],
            expected_output=sample_data.get('expected_output'),
            is_selected=False
        )
        db.add(dataset_sample)
    
    db.commit()
    db.refresh(experiment)
    
    return experiment


@router.get("/", response_model=List[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    """
    List all experiments.
    
    Returns:
        List of all experiments with baseline_metrics
    """
    # ✅ FIX: Eager load baseline_metrics relationship
    experiments = db.query(Experiment)\
        .options(joinedload(Experiment.baseline_metrics))\
        .order_by(Experiment.created_at.desc())\
        .all()
    return experiments


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Get experiment details including dataset samples and baseline metrics.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Experiment details with samples and baseline metrics
        
    Raises:
        404: Experiment not found
    """
    # ✅ FIX: Eager load both samples and baseline_metrics relationships
    experiment = db.query(Experiment)\
        .options(
            joinedload(Experiment.samples),
            joinedload(Experiment.baseline_metrics)
        )\
        .filter(Experiment.id == experiment_id)\
        .first()
    
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found"
        )
    return experiment


@router.delete("/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Delete an experiment and all its samples.
    
    Args:
        experiment_id: ID of the experiment to delete
        
    Raises:
        404: Experiment not found
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found"
        )
    
    db.delete(experiment)
    db.commit()
    return None


@router.post("/{experiment_id}/generate-outputs")
def trigger_generate_outputs(experiment_id: int, db: Session = Depends(get_db)):
    """
    Trigger background task to generate outputs for samples without expected outputs.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Task ID and status
        
    Raises:
        404: Experiment not found
        400: Experiment already has reference outputs
    """
    from app.tasks.generate_outputs import generate_outputs_task
    
    # Validate experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found"
        )
    
    # Check if experiment needs output generation
    if experiment.has_reference_outputs:
        raise HTTPException(
            status_code=400,
            detail="Experiment already has reference outputs"
        )
    
    # Trigger Celery task
    task = generate_outputs_task.delay(experiment_id, sample_limit=20)
    
    return {
        "message": "Output generation task started",
        "task_id": task.id,
        "experiment_id": experiment_id,
        "status": "processing"
    }


@router.post("/{experiment_id}/baseline-evaluation")
def trigger_baseline_evaluation(experiment_id: int, db: Session = Depends(get_db)):
    """
    Trigger baseline evaluation for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Task ID and status
        
    Raises:
        404: Experiment not found
        400: Experiment not ready for evaluation
    """
    from app.tasks.baseline_evaluation import baseline_evaluation_task
    from app.models import BaselineMetrics
    
    # Validate experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found"
        )
    
    # Check if experiment has samples
    sample_count = db.query(DatasetSample).filter(
        DatasetSample.experiment_id == experiment_id
    ).count()
    
    if sample_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Experiment has no samples to evaluate"
        )
    
    # ✅ FIX: Allow re-evaluation, just check if one is currently running
    existing_baseline = db.query(BaselineMetrics).filter(
        BaselineMetrics.experiment_id == experiment_id,
        BaselineMetrics.status == "running"
    ).first()
    
    if existing_baseline:
        raise HTTPException(
            status_code=400,
            detail="Baseline evaluation is already running for this experiment"
        )
    
    # Trigger Celery task
    task = baseline_evaluation_task.delay(experiment_id)
    
    return {
        "message": "Baseline evaluation task started",
        "task_id": task.id,
        "experiment_id": experiment_id,
        "status": "processing"
    }


@router.get("/{experiment_id}/baseline-metrics")
def get_baseline_metrics(experiment_id: int, db: Session = Depends(get_db)):
    """
    Get baseline evaluation metrics for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Baseline metrics
        
    Raises:
        404: Experiment or metrics not found
    """
    from app.models import BaselineMetrics
    from app.schemas import BaselineMetricsResponse
    
    # Validate experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment with ID {experiment_id} not found"
        )
    
    # Get baseline metrics
    metrics = db.query(BaselineMetrics).filter(
        BaselineMetrics.experiment_id == experiment_id
    ).first()
    
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline metrics found for experiment {experiment_id}"
        )
    
    return metrics