"""Dense embedding feature extractor module."""

from typing import List, Optional
import numpy as np


class DenseEmbeddingExtractor:
    """Fallback / placeholder dense embedding feature extractor."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, texts: List[str]):
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into dense representation."""
        try:
            from sentence_transformers import SentenceTransformer
            if self.model is None:
                self.model = SentenceTransformer(self.model_name)
            return self.model.encode(texts, show_progress_bar=False)
        except Exception:
            # Fallback to zeros array if sentence_transformers is not installed
            return np.zeros((len(texts), 128), dtype=float)
