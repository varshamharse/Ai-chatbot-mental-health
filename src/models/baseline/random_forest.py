"""Random Forest classifier for mental health text classification."""

import os
import pickle
from typing import Optional, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """Random Forest ensemble classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
        self.is_fitted = False

    def fit(self, X, y):
        """Fit ensemble model on features and target labels."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probability distribution."""
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: list, top_k: int = 15) -> Dict[str, float]:
        """Get top feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_k]
        return {feature_names[i]: float(importances[i]) for i in sorted_idx}

    def save(self, file_path: str) -> None:
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: str) -> "RandomForestModel":
        """Load model artifact from disk."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self
