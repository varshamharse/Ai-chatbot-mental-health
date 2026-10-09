"""Model Evaluator module orchestrating metrics, confusion matrices, and ROC curves."""

import os
import json
import time
from typing import Dict, Any, Optional, List
import numpy as np

from .metrics import ModelMetricsCalculator
from .confusion_matrix import ConfusionMatrixVisualizer
from .roc_curve import ROCCurveVisualizer


class ModelEvaluator:
    """Standardized evaluation framework for AI Chatbot Mental Health models."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        results_dir: str = "results"
    ):
        self.class_names = class_names or ["Non-Stress", "Stress"]
        self.results_dir = results_dir
        self.cm_visualizer = ConfusionMatrixVisualizer(
            class_names=self.class_names,
            output_dir=os.path.join(results_dir, "confusion_matrices"),
            figures_dir=os.path.join(results_dir, "figures")
        )
        self.roc_visualizer = ROCCurveVisualizer(
            output_dir=os.path.join(results_dir, "roc_curves"),
            figures_dir=os.path.join(results_dir, "figures")
        )

    def evaluate_model(
        self,
        model: Any,
        X_test: Any,
        y_test: np.ndarray,
        model_name: str = "model",
        training_time_sec: float = 0.0,
        dataset_split_name: str = "test"
    ) -> Dict[str, Any]:
        """Perform evaluation of a trained model on a dataset split.
        
        Args:
            model: Trained classifier with predict and optional predict_proba methods.
            X_test: Test feature matrix.
            y_test: Ground truth labels array.
            model_name: Identifier for model (e.g. 'random_forest', 'svm').
            training_time_sec: Training duration in seconds.
            dataset_split_name: Name of split (e.g. 'validation', 'test').
            
        Returns:
            Dictionary containing metrics report and visualization file paths.
        """
        # Inference latency measurement
        start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        total_inf_time = time.perf_counter() - start_time
        num_samples = len(y_test)
        latency_ms_per_sample = (total_inf_time / num_samples) * 1000.0 if num_samples > 0 else 0.0

        # Predicted probabilities
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        # Compute all metrics
        metrics = ModelMetricsCalculator.evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            training_time_sec=training_time_sec,
            inference_time_ms=latency_ms_per_sample
        )

        metrics["model_name"] = model_name
        metrics["dataset_split"] = dataset_split_name
        metrics["num_test_samples"] = int(num_samples)

        # Generate Confusion Matrix
        cm_fig_path = self.cm_visualizer.plot_and_save(
            y_true=y_test,
            y_pred=y_pred,
            model_name=f"{model_name}_{dataset_split_name}",
            normalize=True
        )

        # Generate ROC Curve if probabilities exist
        roc_fig_path = None
        if y_prob is not None:
            roc_fig_path = self.roc_visualizer.plot_roc_curve(
                y_true=y_test,
                y_prob=y_prob,
                model_name=f"{model_name}_{dataset_split_name}"
            )

        report = {
            "metrics": metrics,
            "confusion_matrix_figure": cm_fig_path,
            "roc_curve_figure": roc_fig_path,
        }

        # Save metrics to JSON
        metrics_save_dir = os.path.join(self.results_dir, "metrics")
        os.makedirs(metrics_save_dir, exist_ok=True)
        json_path = os.path.join(metrics_save_dir, f"{model_name}_{dataset_split_name}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
