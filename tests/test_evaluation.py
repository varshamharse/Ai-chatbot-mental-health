"""Unit tests for evaluation metrics, confidence statistics, and confusion matrix."""

import pytest
import numpy as np
from src.evaluation.metrics import ModelMetricsCalculator
from src.evaluation.confusion_matrix import ConfusionMatrixVisualizer
from src.evaluation.model_comparison import ModelComparison


def test_model_metrics_calculator():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.6, 0.4],
        [0.85, 0.15],
        [0.05, 0.95],
        [0.1, 0.9]
    ])

    metrics = ModelMetricsCalculator.evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "mean_confidence" in metrics
    assert "f1_weighted" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_confidence"] <= 1.0
    assert metrics["accuracy"] == 0.875


def test_confidence_metrics():
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2]])
    y_pred = np.array([0, 0])
    res = ModelMetricsCalculator.calculate_confidence_metrics(y_prob, y_pred)
    assert res["mean_confidence"] == 0.85
    assert res["high_confidence_ratio"] == 1.0


def test_confusion_matrix_computation():
    visualizer = ConfusionMatrixVisualizer(class_names=["Non-Stress", "Stress"])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = visualizer.compute(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
