"""Bag-of-Words (CountVectorizer) feature extraction module."""

import os
import pickle
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer


class BoWFeatureExtractor:
    """Bag-of-Words feature extraction."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df
        )

    def fit(self, texts: List[str]):
        """Fit vectorizer on training text corpus."""
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        """Transform texts into token count matrix."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit and transform training texts."""
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return list of vocabulary terms."""
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, file_path: str = "models/vectorizers/bow_vectorizer.pkl") -> None:
        """Save vectorizer to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, file_path: str = "models/vectorizers/bow_vectorizer.pkl") -> "BoWFeatureExtractor":
        """Load vectorizer from disk."""
        with open(file_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        return self
