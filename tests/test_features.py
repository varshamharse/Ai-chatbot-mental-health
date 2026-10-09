"""Unit tests for feature extraction modules."""

import pytest
from src.features.tfidf import TFIDFFeatureExtractor
from src.features.bow import BoWFeatureExtractor
from src.features.semantic_features import SemanticFeatureExtractor


def test_tfidf_extractor():
    texts = [
        "I feel very stressed and overwhelmed",
        "Today is a quiet and peaceful morning",
        "Anxiety and stress are making me exhausted"
    ]
    extractor = TFIDFFeatureExtractor(max_features=20, min_df=1)
    matrix = extractor.fit_transform(texts)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0
    names = extractor.get_feature_names()
    assert len(names) > 0


def test_bow_extractor():
    texts = ["help me please", "i am doing fine"]
    extractor = BoWFeatureExtractor(max_features=10, min_df=1)
    matrix = extractor.fit_transform(texts)
    assert matrix.shape[0] == 2


def test_semantic_feature_extractor():
    texts = ["I feel nervous?", "HELP ME!!!"]
    extractor = SemanticFeatureExtractor()
    feat = extractor.extract_features(texts)
    assert feat.shape == (2, 6)
    # Check question mark in first
    assert feat[0, 3] == 1.0
    # Check exclamation marks in second
    assert feat[1, 4] == 3.0
