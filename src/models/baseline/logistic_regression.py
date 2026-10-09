"""Logistic Regression model wrapper for mental health text classification."""

import os
import pickle
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """Logistic Regression classifier with calibrated probability and weight extraction."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        **kwargs
    ):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.class_weight = class_weight
        self.kwargs = kwargs
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
            class_weight=self.class_weight,
            **self.kwargs
        )
        self.is_fitted = False

    def fit(self, X, y):
        """Fit model on feature matrix and target labels."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probability distribution."""
        return self.model.predict_proba(X)

    def get_top_features(self, feature_names: list, top_k: int = 10) -> Dict[str, list]:
        """Extract top predictive features per class."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        coef = self.model.coef_
        if coef.shape[0] == 1:
            # Binary classification
            top_positive = np.argsort(coef[0])[-top_k:][::-1]
            top_negative = np.argsort(coef[0])[:top_k]
            return {
                "positive_stress": [feature_names[i] for i in top_positive],
                "negative_non_stress": [feature_names[i] for i in top_negative]
            }
        return {}

    def save(self, file_path: str) -> None:
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: str) -> "LogisticRegressionModel":
        """Load model artifact from disk."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self
