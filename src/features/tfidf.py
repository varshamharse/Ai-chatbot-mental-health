"""TF-IDF feature extraction module."""

import os
import pickle
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFFeatureExtractor:
    """TF-IDF text feature extraction and vectorization."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_df=self.max_df
        )

    def fit(self, texts: List[str]):
        """Fit vectorizer on training text corpus."""
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        """Transform texts into sparse TF-IDF matrix."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit and transform training texts."""
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return list of learned vocabulary tokens."""
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, file_path: str = "models/vectorizers/tfidf_vectorizer.pkl") -> None:
        """Save fitted vectorizer to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, file_path: str = "models/vectorizers/tfidf_vectorizer.pkl") -> "TFIDFFeatureExtractor":
        """Load fitted vectorizer from disk."""
        with open(file_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        return self
