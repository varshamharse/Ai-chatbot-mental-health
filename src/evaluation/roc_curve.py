"""ROC and Precision-Recall curve calculation and plotting module."""

import os
from typing import Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class ROCCurveVisualizer:
    """Computes and plots ROC curves and Precision-Recall curves."""

    def __init__(
        self,
        output_dir: str = "results/roc_curves",
        figures_dir: str = "results/figures"
    ):
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str
    ) -> Optional[str]:
        """Compute and save ROC curve plot."""
        if y_prob is None or len(np.unique(y_true)) != 2:
            return None

        pos_probs = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob.ravel()
        fpr, tpr, _ = roc_curve(y_true, pos_probs)
        roc_auc = auc(fpr, tpr)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 5), dpi=300)
            plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
            plt.title(f"{model_name.replace('_', ' ').title()} — ROC Curve", fontsize=12, pad=12)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            fig_path = os.path.join(self.figures_dir, f"{model_name}_roc_curve.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            return fig_path
        except Exception:
            return None
