"""Comprehensive metric calculation module for mental health model evaluation."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    brier_score_loss
)


class ModelMetricsCalculator:
    """Calculates accuracy, confidence, specificity, precision, recall, f1, and ROC-AUC."""

    @staticmethod
    def calculate_confidence_metrics(
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.75
    ) -> Dict[str, float]:
        """Compute predicted confidence statistics.
        
        Args:
            y_prob: Predicted probability matrix of shape (n_samples, n_classes).
            y_pred: Predicted class labels.
            y_true: Optional ground truth labels.
            confidence_threshold: Threshold for high confidence calculation.
            
        Returns:
            Dictionary of confidence metrics.
        """
        if y_prob is None or len(y_prob) == 0:
            return {
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "std_confidence": 0.0,
                "high_confidence_ratio": 0.0
            }

        max_probs = np.max(y_prob, axis=1)
        mean_conf = float(np.mean(max_probs))
        median_conf = float(np.median(max_probs))
        std_conf = float(np.std(max_probs))
        high_conf_ratio = float(np.mean(max_probs >= confidence_threshold))

        res = {
            "mean_confidence": round(mean_conf, 4),
            "median_confidence": round(median_conf, 4),
            "std_confidence": round(std_conf, 4),
            "high_confidence_ratio": round(high_conf_ratio, 4)
        }

        # If y_true is provided, measure confidence when prediction is correct vs incorrect
        if y_true is not None:
            correct_mask = (y_pred == y_true)
            if np.sum(correct_mask) > 0:
                res["correct_predictions_mean_confidence"] = round(float(np.mean(max_probs[correct_mask])), 4)
            if np.sum(~correct_mask) > 0:
                res["incorrect_predictions_mean_confidence"] = round(float(np.mean(max_probs[~correct_mask])), 4)

        return res

    @staticmethod
    def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate) for binary classification."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            # Multiclass macro specificity
            specificities = []
            for i in range(cm.shape[0]):
                tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            return float(np.mean(specificities))

    @classmethod
    def evaluate_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        training_time_sec: float = 0.0,
        inference_time_ms: float = 0.0
    ) -> Dict[str, Any]:
        """Compute full set of classification metrics.
        
        Args:
            y_true: Ground truth target labels.
            y_pred: Predicted target labels.
            y_prob: Predicted probability array.
            training_time_sec: Time taken for model training.
            inference_time_ms: Latency per sample in milliseconds.
            
        Returns:
            Dictionary with all computed performance metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        specificity = cls.calculate_specificity(y_true, y_pred)

        # ROC-AUC & Log Loss
        roc_auc = None
        logloss = None
        if y_prob is not None:
            try:
                unique_classes = np.unique(y_true)
                if len(unique_classes) == 2:
                    pos_probs = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
                    roc_auc = float(roc_auc_score(y_true, pos_probs))
                else:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc = None

            try:
                logloss = float(log_loss(y_true, y_prob))
            except Exception:
                logloss = None

        # Confidence Metrics
        conf_metrics = cls.calculate_confidence_metrics(y_prob, y_pred, y_true) if y_prob is not None else {}

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "recall_macro": round(float(rec_macro), 4),
            "recall_weighted": round(float(rec_weighted), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "specificity": round(float(specificity), 4),
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
            "log_loss": round(float(logloss), 4) if logloss is not None else "N/A",
            "mean_confidence": conf_metrics.get("mean_confidence", 0.0),
            "median_confidence": conf_metrics.get("median_confidence", 0.0),
            "high_confidence_ratio": conf_metrics.get("high_confidence_ratio", 0.0),
            "training_time_sec": round(float(training_time_sec), 4),
            "inference_time_ms_per_sample": round(float(inference_time_ms), 4)
        }

        return metrics
