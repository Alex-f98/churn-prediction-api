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
        logger.info({
                        "event": "prediction",
                        "probability": float(result["churn_probability"]),
                        "model_version": predictor.version,
                        "input_sample": data.dict()
                    })
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/stats")
def stats():
    return {
        "total_predictions": predictor.request_count
    }

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/metadata")
def metadata():
    return {
        "model_version": predictor.version,
        "features": list(CustomerInput.model_fields.keys()),
        "target": "churn",
        "description": "Customer churn prediction model",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metadata": "/metadata",
            "stats": "/stats"
        }

    }