from fastapi import APIRouter, HTTPException
from app.schemas import CustomerInput
from service.predictor import Predictor
from app.logger import logger


router = APIRouter()
predictor = Predictor()

@router.post("/predict")
def predict(data: CustomerInput):
    try:
        result = predictor.predict(data.dict())
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/health")
def health():
    return {"status": "ok"}