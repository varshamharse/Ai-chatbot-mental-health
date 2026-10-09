"""Script to train Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes models.

Usage:
    python scripts/train.py
    python scripts/train.py --models random_forest svm logistic_regression knn naive_bayes
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_baseline import BaselineTrainer
from src.data.loader import DatasetLoader
from src.data.splitter import DatasetSplitter
from src.preprocessing.pipeline import TextPreprocessingPipeline
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_script")


def ensure_processed_data() -> tuple:
    """Load processed splits, or prepare them on the fly if not yet present."""
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        return train_df, val_df

    logger.info("Processed data not found. Preparing from raw dataset...")
    loader = DatasetLoader(config_path="configs/dataset.yaml")
    df = loader.load_dataset()
    schema = loader.config.get("schema", {})
    text_col = schema.get("text_column", "text")
    label_col = schema.get("label_column", "label")

    df_clean = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    pipeline = TextPreprocessingPipeline(config_path="configs/preprocessing.yaml")
    df_clean["cleaned_text"] = pipeline.transform(df_clean[text_col])
    df_clean = df_clean[df_clean["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

    splitter = DatasetSplitter.from_config(loader.config)
    train_df, val_df, test_df = splitter.split(df_clean)
    splitter.save_splits(train_df, val_df, test_df)

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Train mental health baseline classification models.")
    parser.add_argument("--config", type=str, default="configs/models.yaml", help="Path to models config.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of models to train.")
    args = parser.parse_args()

    set_seed(42)
    logger.info("Starting baseline model training pipeline...")

    # 1. Load Data
    train_df, val_df = ensure_processed_data()
    text_col = "cleaned_text" if "cleaned_text" in train_df.columns else "text"
    label_col = "label"

    train_texts = train_df[text_col].astype(str).tolist()
    train_labels = train_df[label_col].to_numpy()
    val_texts = val_df[text_col].astype(str).tolist()
    val_labels = val_df[label_col].to_numpy()

    logger.info(f"Loaded {len(train_texts)} training samples and {len(val_texts)} validation samples.")

    # 2. Train Models
    trainer = BaselineTrainer(config_path=args.config)
    results = trainer.train_all(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels
    )

    # 3. Print Results Summary
    print("\n" + "="*60)
    print("MODEL TRAINING & VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Val F1':<10} | {'Confidence':<10}")
    print("-" * 60)
    for report in results["validation_reports"]:
        m = report["metrics"]
        print(f"{m['model_name']:<22} | {m['accuracy']:<10.4f} | {m['f1_weighted']:<10.4f} | {m['mean_confidence']:<10.4f}")
    print("-" * 60)
    print(f"Best Model Selected: {results['best_model_name']}")
    print(f"Checkpoints directory: models/checkpoints/")
    print(f"Best model directory:  models/best_model/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
