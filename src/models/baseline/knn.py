"""K-Nearest Neighbors (KNN) classifier for mental health text classification."""

import os
import pickle
from typing import Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNModel:
    """K-Nearest Neighbors classifier with distance weighting and probability prediction."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "distance",
        metric: str = "cosine",
        n_jobs: int = -1,
        **kwargs
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
        self.is_fitted = False

    def fit(self, X, y):
        """Fit KNN model on features and target labels."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probability distribution."""
        return self.model.predict_proba(X)

    def save(self, file_path: str) -> None:
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: str) -> "KNNModel":
        """Load model artifact from disk."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self
