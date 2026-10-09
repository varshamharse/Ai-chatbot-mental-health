"""Feature engineering and extraction modules."""

from .tfidf import TFIDFFeatureExtractor
from .bow import BoWFeatureExtractor
from .semantic_features import SemanticFeatureExtractor
from .embeddings import DenseEmbeddingExtractor

__all__ = [
    "TFIDFFeatureExtractor",
    "BoWFeatureExtractor",
    "SemanticFeatureExtractor",
    "DenseEmbeddingExtractor"
]
