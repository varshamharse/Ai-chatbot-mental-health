"""Inference predictor for classifying mental health and stress states from raw text."""

import os
import time
import pickle
from typing import Dict, Any, Optional, List
import numpy as np

from src.preprocessing.pipeline import TextPreprocessingPipeline
from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.file_utils import load_yaml, load_model_artifact, load_json


class MentalHealthPredictor:
    """End-to-end predictor: Raw text -> Preprocessing -> Feature Extraction -> Model Inference."""

    def __init__(
        self,
        model_path: str = "models/best_model/best_model.pkl",
        vectorizer_path: str = "models/vectorizers/tfidf_vectorizer.pkl",
        config_path: str = "configs/dataset.yaml"
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.config_path = config_path

        self.pipeline = TextPreprocessingPipeline(config_path="configs/preprocessing.yaml")
        self.model = None
        self.vectorizer = None
        self.classes_map = {0: "Non-Stress", 1: "Stress"}
        self._load_components()

    def _load_components(self) -> None:
        """Load trained model, vectorizer, and class mapping."""
        # Load vectorizer
        if os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

        # Load model
        target_model = self.model_path
        if not os.path.exists(target_model):
            # Check checkpoints fallback
            ckpts = ["models/checkpoints/logistic_regression.pkl", "models/checkpoints/svm.pkl", "models/checkpoints/random_forest.pkl"]
            for c in ckpts:
                if os.path.exists(c):
                    target_model = c
                    break

        if os.path.exists(target_model):
            with open(target_model, "rb") as f:
                self.model = pickle.load(f)

        # Load class schema from config if available
        if os.path.exists(self.config_path):
            cfg = load_yaml(self.config_path)
            classes_cfg = cfg.get("task", {}).get("primary", {}).get("classes", {})
            if classes_cfg:
                self.classes_map = {int(k): str(v) for k, v in classes_cfg.items()}

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify single user input text and return prediction with confidence.
        
        Args:
            text: Raw input string from user.
            
        Returns:
            Dictionary with predicted class, numeric label, confidence score, and latency.
        """
        if self.model is None or self.vectorizer is None:
            # Fallback for uninitialized models
            return {
                "input": text,
                "cleaned_text": self.pipeline.preprocess_text(text),
                "predicted_label": 0,
                "predicted_class": "Non-Stress",
                "confidence": 0.50,
                "probabilities": {"Non-Stress": 0.50, "Stress": 0.50},
                "latency_ms": 0.0,
                "model_status": "model_not_trained_yet"
            }

        t0 = time.perf_counter()
        cleaned_text = self.pipeline.preprocess_text(text)
        features = self.vectorizer.transform([cleaned_text])

        # Predict
        y_pred = int(self.model.predict(features)[0])
        probabilities = {}
        confidence = 0.5

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = float(np.max(probs))
            for idx, p in enumerate(probs):
                cls_name = self.classes_map.get(idx, f"Class_{idx}")
                probabilities[cls_name] = round(float(p), 4)
        else:
            confidence = 1.0
            probabilities = {self.classes_map.get(y_pred, str(y_pred)): 1.0}

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "input": text,
            "cleaned_text": cleaned_text,
            "predicted_label": y_pred,
            "predicted_class": self.classes_map.get(y_pred, str(y_pred)),
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2),
            "model_status": "active"
        }
