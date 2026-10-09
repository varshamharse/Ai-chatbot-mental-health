"""API router defining health, predict, and chat endpoints."""

import os
from fastapi import APIRouter, HTTPException
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse, ChatRequest, ChatResponse
from src.inference.chatbot_engine import MentalHealthChatbotEngine
from src.inference.predictor import MentalHealthPredictor
from chatbot.safety.escalation import CrisisEscalator
from src.utils.file_utils import load_json

router = APIRouter()

# Initialize chatbot engine and predictor
engine = MentalHealthChatbotEngine()
predictor = MentalHealthPredictor()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """System health check endpoint."""
    is_loaded = engine.predictor.model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        version="1.0.0"
    )


@router.post("/predict", response_model=PredictResponse)
def predict_mental_health_state(req: PredictRequest):
    """Predict mental health category (Stress vs Non-Stress) with confidence score."""
    try:
        res = predictor.predict(req.text)
        return PredictResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat_with_bot(req: ChatRequest):
    """Full conversational endpoint with crisis detection, ML classification, and empathetic reply."""
    try:
        res = engine.process_message(req.message)
        return ChatResponse(
            response=res["response"],
            predicted_class=res["predicted_class"],
            confidence=res["confidence"],
            is_crisis=res["is_crisis"],
            risk_level=res["risk_level"],
            disclaimer=res["disclaimer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
def get_model_information():
    """Return metadata about the current deployed best model."""
    meta_path = "models/best_model/metadata.json"
    if os.path.exists(meta_path):
        return load_json(meta_path)
    return {"status": "default_model", "info": "Model checkpoints available under models/checkpoints/"}


@router.get("/hotlines")
def get_crisis_hotlines():
    """Return crisis emergency contact directory."""
    return CrisisEscalator.get_all_hotlines()
