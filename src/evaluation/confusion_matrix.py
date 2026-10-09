"""Confusion Matrix computation and visualization module."""

import os
import json
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class ConfusionMatrixVisualizer:
    """Computes, exports, and plots research-quality confusion matrices."""

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: str = "results/confusion_matrices",
        figures_dir: str = "results/figures"
    ):
        self.class_names = class_names or ["Non-Stress", "Stress"]
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute raw count confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def save_matrix_csv(self, cm: np.ndarray, model_name: str) -> str:
        """Save confusion matrix as a formatted CSV file."""
        df_cm = pd.DataFrame(
            cm,
            index=[f"Actual_{c}" for c in self.class_names[:cm.shape[0]]],
            columns=[f"Pred_{c}" for c in self.class_names[:cm.shape[1]]]
        )
        csv_path = os.path.join(self.output_dir, f"{model_name}_confusion_matrix.csv")
        df_cm.to_csv(csv_path)
        return csv_path

    def plot_and_save(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        normalize: bool = True
    ) -> Optional[str]:
        """Plot confusion matrix heatmap and save high-resolution figure."""
        cm = self.compute(y_true, y_pred)
        self.save_matrix_csv(cm, model_name)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
            fmt = ".2%" if normalize else "d"

            plt.figure(figsize=(6, 5), dpi=300)
            sns.heatmap(
                cm_display,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=self.class_names[:cm.shape[1]],
                yticklabels=self.class_names[:cm.shape[0]],
                cbar=True
            )
            plt.title(f"{model_name.replace('_', ' ').title()} — Confusion Matrix", fontsize=12, pad=12)
            plt.xlabel("Predicted Class", fontsize=10)
            plt.ylabel("Actual Class", fontsize=10)
            plt.tight_layout()

            fig_path = os.path.join(self.figures_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            return fig_path
        except Exception:
            return None
