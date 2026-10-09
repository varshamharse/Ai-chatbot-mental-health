"""Support Vector Machine (SVM) classifier for mental health text classification."""

import os
import pickle
from typing import Optional
import numpy as np
from sklearn.svm import SVC


class SVMModel:
    """Support Vector Machine classifier with probability estimates."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        probability: bool = True,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        **kwargs
    ):
        self.C = C
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.class_weight = class_weight
        self.kwargs = kwargs
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
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
        if self.probability:
            return self.model.predict_proba(X)
        else:
            # Fallback if probability wasn't enabled: apply sigmoid to decision function
            df = self.model.decision_function(X)
            probs = 1 / (1 + np.exp(-df))
            return np.vstack([1 - probs, probs]).T

    def decision_function(self, X) -> np.ndarray:
        """Evaluate decision function."""
        return self.model.decision_function(X)

    def save(self, file_path: str) -> None:
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: str) -> "SVMModel":
        """Load model artifact from disk."""
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self
