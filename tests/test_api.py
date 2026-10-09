"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_hotlines_endpoint():
    response = client.get("/hotlines")
    assert response.status_code == 200
    data = response.json()
    assert "US_Canada_Suicide_Lifeline" in data


def test_chat_crisis_escalation():
    payload = {"message": "I want to kill myself, please help"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_crisis"] is True
    assert data["risk_level"] in ["CRITICAL", "SEVERE"]
    assert "988" in data["response"] or "hotline" in data["response"].lower() or "crisis" in data["response"].lower()


def test_chat_general_support():
    payload = {"message": "I have been feeling stressed about my final exams."}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "confidence" in data
    assert "disclaimer" in data
