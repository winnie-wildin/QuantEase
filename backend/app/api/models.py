from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import Model
from app.schemas import ModelResponse

router = APIRouter()
print("✅ DEBUG: Models router created")


@router.get("/", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    """
    List all available pre-supported models.
    
    Returns:
        List of available models (GPT-2, Llama-2-7B, Mistral-7B)
    """
    print("✅ DEBUG: list_models endpoint called")
    models = db.query(Model).all()
    return models


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """
    Get a specific model by ID.
    
    Args:
        model_id: ID of the model to retrieve
        
    Returns:
        Model details
        
    Raises:
        404: Model not found
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    return model