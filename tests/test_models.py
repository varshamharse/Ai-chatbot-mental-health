"""Unit tests for baseline classification models."""

import pytest
import numpy as np
from src.models.baseline import (
    LogisticRegressionModel,
    NaiveBayesModel,
    SVMModel,
    RandomForestModel,
    KNNModel
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.rand(40, 10)
    y = np.random.choice([0, 1], size=40)
    return X, y


def test_logistic_regression(dummy_data):
    X, y = dummy_data
    model = LogisticRegressionModel(C=1.0, max_iter=200, random_state=42)
    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])
    assert len(preds) == 5
    assert probs.shape == (5, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_svm_model(dummy_data):
    X, y = dummy_data
    model = SVMModel(kernel="linear", probability=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])
    assert len(preds) == 5
    assert probs.shape == (5, 2)


def test_random_forest_model(dummy_data):
    X, y = dummy_data
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])
    assert len(preds) == 5
    assert probs.shape == (5, 2)


def test_knn_model(dummy_data):
    X, y = dummy_data
    model = KNNModel(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])
    assert len(preds) == 5
    assert probs.shape == (5, 2)


def test_naive_bayes_model(dummy_data):
    X, y = dummy_data
    model = NaiveBayesModel(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])
    assert len(preds) == 5
    assert probs.shape == (5, 2)
