"""Feature extraction helpers for text preprocessing pipeline."""

from typing import Dict, Any, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:
    """Extracts numerical matrices (TF-IDF, BoW) from preprocessed text collections."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = tuple(ngram_range)
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = self._init_vectorizer()

    def _init_vectorizer(self):
        if self.method == "tfidf":
            return TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
                min_df=self.min_df,
                max_df=self.max_df
            )
        elif self.method == "bow":
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        else:
            raise ValueError(f"Unsupported extraction method: {self.method}")

    def fit(self, texts: List[str]):
        """Fit vectorizer on training text corpus."""
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        """Transform texts into numerical sparse feature matrix."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Fit on texts and transform them."""
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return list of vocabulary feature terms."""
        return self.vectorizer.get_feature_names_out().tolist()
