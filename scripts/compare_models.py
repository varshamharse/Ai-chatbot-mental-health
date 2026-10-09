"""Script to compile model comparison table and research report.

Usage:
    python scripts/compare_models.py
"""

import os
import sys
import json
import argparse
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.model_comparison import ModelComparison
from src.utils.logger import get_logger

logger = get_logger("compare_script")


def main():
    parser = argparse.ArgumentParser(description="Compile model comparison report.")
    parser.add_argument("--metrics_dir", type=str, default="results/metrics", help="Directory with JSON metric reports.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save final comparison table.")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_dir):
        logger.error(f"Metrics directory '{args.metrics_dir}' not found. Please run scripts/evaluate.py first.")
        sys.exit(1)

    metric_files = [f for f in os.listdir(args.metrics_dir) if f.endswith(".json") and "test" in f]
    if not metric_files:
        logger.warning(f"No test metric reports found in {args.metrics_dir}. Checking all json files...")
        metric_files = [f for f in os.listdir(args.metrics_dir) if f.endswith(".json")]

    if not metric_files:
        logger.error("No metric files found to compare.")
        sys.exit(1)

    reports = []
    for mf in metric_files:
        with open(os.path.join(args.metrics_dir, mf), "r", encoding="utf-8") as f:
            reports.append(json.load(f))

    comparator = ModelComparison(results_dir=args.results_dir)
    comp_df = comparator.generate_comparison_table(reports, primary_metric="f1_weighted")
    charts = comparator.plot_comparison_charts(comp_df)
    best_model = comparator.get_best_model(comp_df)

    print("\n" + "="*80)
    print("MSc AI RESEARCH — FINAL MODEL BENCHMARK & COMPARISON")
    print("="*80)
    print(comp_df.to_string(index=False))
    print("-" * 80)
    print(f"🏆 BEST PERFORMING MODEL: {best_model.get('Model', 'N/A')}")
    print(f"   Accuracy:        {best_model.get('Accuracy')}")
    print(f"   F1-Score:        {best_model.get('F1_Score')}")
    print(f"   Confidence:      {best_model.get('Confidence')}")
    print(f"   Specificity:     {best_model.get('Specificity')}")
    print(f"   ROC-AUC:         {best_model.get('ROC_AUC')}")
    print(f"   Inference Time:  {best_model.get('Inference_Time_ms')} ms")
    print("="*80)
    print(f"Results exported to {os.path.join(args.results_dir, 'model_comparison.csv')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
