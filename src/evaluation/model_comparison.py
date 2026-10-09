"""Model comparison and ranking module for AI Chatbot Mental Health."""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class ModelComparison:
    """Aggregates metrics across multiple models, generates comparison tables, and plots comparative figures."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def generate_comparison_table(
        self,
        eval_reports: List[Dict[str, Any]],
        primary_metric: str = "f1_weighted"
    ) -> pd.DataFrame:
        """Create comparison DataFrame from list of evaluation reports.
        
        Args:
            eval_reports: List of reports returned from ModelEvaluator.
            primary_metric: Metric used to rank models.
            
        Returns:
            pd.DataFrame: Ranked comparison table.
        """
        rows = []
        for r in eval_reports:
            m = r.get("metrics", r)
            rows.append({
                "Model": m.get("model_name", "Unknown").replace("_", " ").title(),
                "Accuracy": m.get("accuracy", 0.0),
                "Confidence": m.get("mean_confidence", 0.0),
                "Precision": m.get("precision_weighted", 0.0),
                "Recall": m.get("recall_weighted", 0.0),
                "F1_Score": m.get("f1_weighted", 0.0),
                "Specificity": m.get("specificity", 0.0),
                "ROC_AUC": m.get("roc_auc", "N/A"),
                "Training_Time_s": m.get("training_time_sec", 0.0),
                "Inference_Time_ms": m.get("inference_time_ms_per_sample", 0.0)
            })

        df = pd.DataFrame(rows)
        # Sort by primary metric if numeric
        sort_col = "F1_Score" if "f1" in primary_metric.lower() else "Accuracy"
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

        # Export to results/model_comparison.csv
        csv_path = os.path.join(self.results_dir, "model_comparison.csv")
        df.to_csv(csv_path, index=False)

        # Export final_report.csv
        report_path = os.path.join(self.results_dir, "final_report.csv")
        df.to_csv(report_path, index=False)

        return df

    def plot_comparison_charts(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate research-quality comparison bar plots."""
        saved_figures = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 1. Accuracy & F1 Comparison Plot
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            x = np.arange(len(comparison_df))
            width = 0.35

            rects1 = ax.bar(x - width/2, comparison_df["Accuracy"], width, label="Accuracy", color="#3b82f6")
            rects2 = ax.bar(x + width/2, comparison_df["F1_Score"], width, label="F1-Score", color="#10b981")

            ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
            ax.set_title("Performance Comparison Across Baseline Models", fontsize=13, pad=14)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            # Add value labels on bars
            for rect in rects1:
                h = rect.get_height()
                ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
            for rect in rects2:
                h = rect.get_height()
                ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            perf_chart_path = os.path.join(self.figures_dir, "model_performance_comparison.png")
            plt.savefig(perf_chart_path, dpi=300)
            plt.close()
            saved_figures.append(perf_chart_path)

            # 2. Confidence Comparison Plot
            if "Confidence" in comparison_df.columns:
                plt.figure(figsize=(8, 5), dpi=300)
                sns.barplot(data=comparison_df, x="Model", y="Confidence", palette="viridis")
                plt.title("Model Prediction Confidence Comparison", fontsize=12, pad=12)
                plt.ylabel("Mean Prediction Confidence", fontsize=10)
                plt.ylim(0, 1.05)
                plt.xticks(rotation=15, ha="right")
                plt.grid(axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                conf_chart_path = os.path.join(self.figures_dir, "model_confidence_comparison.png")
                plt.savefig(conf_chart_path, dpi=300)
                plt.close()
                saved_figures.append(conf_chart_path)

        except Exception:
            pass

        return saved_figures

    def get_best_model(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify the best performing model."""
        if comparison_df.empty:
            return {}
        best_row = comparison_df.iloc[0].to_dict()
        return best_row
