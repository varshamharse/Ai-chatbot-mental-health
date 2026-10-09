"""Pydantic schemas for FastAPI mental health chatbot backend."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="I feel so stressed and anxious about my upcoming exams.")


class PredictResponse(BaseModel):
    input: str
    cleaned_text: str
    predicted_label: int
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    model_status: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, example="I've been feeling overwhelmed and cannot sleep.")
    session_id: Optional[str] = Field("default_session", example="session_123")


class ChatResponse(BaseModel):
    response: str
    predicted_class: str
    confidence: float
    is_crisis: bool
    risk_level: str
    disclaimer: str
