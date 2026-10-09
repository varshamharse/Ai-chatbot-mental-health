"""Script to clean, preprocess, and partition dataset into train/validation/test splits.

Usage:
    python scripts/prepare_dataset.py --config configs/dataset.yaml
"""

import os
import sys
import argparse
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator
from src.data.splitter import DatasetSplitter
from src.preprocessing.pipeline import TextPreprocessingPipeline
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("prepare_dataset")


def main():
    parser = argparse.ArgumentParser(description="Clean, preprocess, and split mental health dataset.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset config file.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed splits.")
    args = parser.parse_args()

    set_seed(42)
    logger.info("Initializing dataset preparation...")

    # 1. Load dataset
    loader = DatasetLoader(config_path=args.config)
    df_raw = loader.load_dataset()
    logger.info(f"Loaded raw dataset with {len(df_raw)} records and {df_raw.shape[1]} columns.")

    # 2. Extract schema
    schema = loader.config.get("schema", {})
    text_col = schema.get("text_column", "text")
    label_col = schema.get("label_column", "label")

    # 3. Clean raw records (missing text, missing labels)
    initial_count = len(df_raw)
    df_clean = df_raw.dropna(subset=[text_col, label_col]).copy()
    
    # Filter empty or whitespace-only texts
    df_clean = df_clean[df_clean[text_col].astype(str).str.strip().str.len() > 3]

    # Deduplicate on text to eliminate data leakage across splits
    df_clean = df_clean.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    dedup_count = len(df_clean)
    logger.info(f"Deduplicated and filtered dataset: {initial_count} -> {dedup_count} records ({initial_count - dedup_count} removed).")

    # 4. Text Preprocessing
    logger.info("Executing NLP cleaning and text normalization pipeline...")
    pipeline = TextPreprocessingPipeline(config_path="configs/preprocessing.yaml")
    df_clean["cleaned_text"] = pipeline.transform(df_clean[text_col])

    # Ensure no empty preprocessed strings remain
    df_clean = df_clean[df_clean["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

    # 5. Stratified Dataset Splitting (70% Train, 15% Val, 15% Test)
    logger.info("Performing stratified train/val/test splitting...")
    splitter = DatasetSplitter.from_config(loader.config)
    train_df, val_df, test_df = splitter.split(df_clean)

    # 6. Save processed splits
    os.makedirs(args.output_dir, exist_ok=True)
    split_paths = splitter.save_splits(train_df, val_df, test_df, output_dir=args.output_dir)

    # 7. Generate preprocessing summary
    os.makedirs("results", exist_ok=True)
    summary_data = [
        {"metric": "Raw Records", "value": initial_count},
        {"metric": "Cleaned & Deduplicated Records", "value": len(df_clean)},
        {"metric": "Training Samples (70%)", "value": len(train_df)},
        {"metric": "Validation Samples (15%)", "value": len(val_df)},
        {"metric": "Test Samples (15%)", "value": len(test_df)},
        {"metric": "Train Stress Count (Class 1)", "value": int((train_df[label_col] == 1).sum())},
        {"metric": "Train Non-Stress Count (Class 0)", "value": int((train_df[label_col] == 0).sum())},
        {"metric": "Test Stress Count (Class 1)", "value": int((test_df[label_col] == 1).sum())},
        {"metric": "Test Non-Stress Count (Class 0)", "value": int((test_df[label_col] == 0).sum())},
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_path = "results/preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved preprocessing summary to {summary_path}")

    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETE")
    print("="*50)
    print(f"Train split: {split_paths['train']} ({len(train_df)} samples)")
    print(f"Val split:   {split_paths['val']} ({len(val_df)} samples)")
    print(f"Test split:  {split_paths['test']} ({len(test_df)} samples)")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
