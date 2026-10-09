"""Script to evaluate trained models on the held-out test set.

Usage:
    python scripts/evaluate.py
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.model_comparison import ModelComparison
from src.features.tfidf import TFIDFFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("evaluate_script")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set.")
    parser.add_argument("--test_data", type=str, default="data/processed/test.csv", help="Path to test set CSV.")
    parser.add_argument("--checkpoints_dir", type=str, default="models/checkpoints", help="Directory with trained model checkpoints.")
    parser.add_argument("--vectorizer_path", type=str, default="models/vectorizers/tfidf_vectorizer.pkl", help="Path to fitted TF-IDF vectorizer.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for evaluation results.")
    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        logger.error(f"Test data file not found at: {args.test_data}. Please run scripts/prepare_dataset.py first.")
        sys.exit(1)

    if not os.path.exists(args.vectorizer_path):
        logger.error(f"Fitted vectorizer not found at: {args.vectorizer_path}. Please run scripts/train.py first.")
        sys.exit(1)

    # 1. Load test data
    test_df = pd.read_csv(args.test_data)
    text_col = "cleaned_text" if "cleaned_text" in test_df.columns else "text"
    test_texts = test_df[text_col].astype(str).tolist()
    y_test = test_df["label"].to_numpy()
    logger.info(f"Loaded {len(test_texts)} test samples.")

    # 2. Load vectorizer and transform test data
    with open(args.vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_test = vectorizer.transform(test_texts)
    logger.info("Transformed test data with TF-IDF vectorizer.")

    # 3. Find and evaluate all checkpoints
    model_files = [f for f in os.listdir(args.checkpoints_dir) if f.endswith(".pkl")]
    if not model_files:
        logger.error(f"No model checkpoints found in {args.checkpoints_dir}. Please run scripts/train.py first.")
        sys.exit(1)

    evaluator = ModelEvaluator(class_names=["Non-Stress", "Stress"], results_dir=args.results_dir)
    reports = []

    print("\n" + "="*80)
    print("EVALUATING MODELS ON HELD-OUT TEST SET")
    print("="*80)

    for mf in sorted(model_files):
        model_name = os.path.splitext(mf)[0]
        model_path = os.path.join(args.checkpoints_dir, mf)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Evaluating {model_name}...")
        report = evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            dataset_split_name="test"
        )
        reports.append(report)

        m = report["metrics"]
        print(f"[{model_name.upper()}]")
        print(f"  • Accuracy:        {m['accuracy']:.4f}")
        print(f"  • Mean Confidence: {m['mean_confidence']:.4f}")
        print(f"  • Precision (Wtd): {m['precision_weighted']:.4f}")
        print(f"  • Recall (Wtd):    {m['recall_weighted']:.4f}")
        print(f"  • F1-Score (Wtd):  {m['f1_weighted']:.4f}")
        print(f"  • Specificity:     {m['specificity']:.4f}")
        print(f"  • ROC-AUC:         {m['roc_auc']}")
        print(f"  • Inference (ms):  {m['inference_time_ms_per_sample']:.2f} ms/sample")
        print("-" * 80)

    # 4. Generate comparison table & plots
    comparator = ModelComparison(results_dir=args.results_dir)
    comp_df = comparator.generate_comparison_table(reports, primary_metric="f1_weighted")
    comparator.plot_comparison_charts(comp_df)

    print("\nMODEL TEST COMPARISON TABLE (Ranked by F1-Score):")
    print(comp_df.to_string(index=False))
    print("\n" + "="*80)
    print(f"Saved comparison table to: {os.path.join(args.results_dir, 'model_comparison.csv')}")
    print(f"Saved confusion matrices to: {os.path.join(args.results_dir, 'confusion_matrices/')}")
    print(f"Saved figures to: {os.path.join(args.results_dir, 'figures/')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
