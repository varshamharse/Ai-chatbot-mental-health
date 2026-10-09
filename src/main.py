"""Master pipeline execution script for AI Chatbot Mental Health project.

Orchestrates:
1. Dataset Loading, Cleaning & Preprocessing
2. Stratified Splitting
3. Baseline Models Training (Random Forest, SVM, Logistic Regression, KNN, Naive Bayes)
4. Comprehensive Test Evaluation (Accuracy, Confidence, Confusion Matrix, ROC-AUC, Specificity)
5. Model Comparison & Best Model Selection
6. Chatbot Verification

Usage:
    python src/main.py
"""

import os
import sys
import argparse
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DatasetLoader
from src.data.splitter import DatasetSplitter
from src.preprocessing.pipeline import TextPreprocessingPipeline
from src.training.train_baseline import BaselineTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.model_comparison import ModelComparison
from src.inference.chatbot_engine import MentalHealthChatbotEngine
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("master_pipeline")


def run_pipeline():
    set_seed(42)
    print("\n" + "="*80)
    print("AI CHATBOT FOR MENTAL HEALTH — END-TO-END RESEARCH PIPELINE")
    print("="*80)

    # Step 1: Preprocess Dataset
    logger.info("Step 1: Loading and Preprocessing Dataset...")
    loader = DatasetLoader(config_path="configs/dataset.yaml")
    df_raw = loader.load_dataset()
    schema = loader.config.get("schema", {})
    text_col = schema.get("text_column", "text")
    label_col = schema.get("label_column", "label")

    df_clean = df_raw.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    pipeline = TextPreprocessingPipeline(config_path="configs/preprocessing.yaml")
    df_clean["cleaned_text"] = pipeline.transform(df_clean[text_col])
    df_clean = df_clean[df_clean["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)

    # Step 2: Stratified Split
    logger.info("Step 2: Splitting Dataset (70% Train, 15% Val, 15% Test)...")
    splitter = DatasetSplitter.from_config(loader.config)
    train_df, val_df, test_df = splitter.split(df_clean)
    splitter.save_splits(train_df, val_df, test_df)

    train_texts = train_df["cleaned_text"].tolist()
    train_labels = train_df[label_col].to_numpy()
    val_texts = val_df["cleaned_text"].tolist()
    val_labels = val_df[label_col].to_numpy()
    test_texts = test_df["cleaned_text"].tolist()
    test_labels = test_df[label_col].to_numpy()

    # Step 3: Train 5 Models
    logger.info("Step 3: Training Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes...")
    trainer = BaselineTrainer(config_path="configs/models.yaml")
    train_results = trainer.train_all(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels
    )
    vectorizer = train_results["vectorizer"]
    trained_models = train_results["trained_models"]

    # Step 4: Evaluate on Held-out Test Set
    logger.info("Step 4: Evaluating on Held-out Test Set...")
    X_test = vectorizer.transform(test_texts)
    evaluator = ModelEvaluator(class_names=["Non-Stress", "Stress"], results_dir="results")
    test_reports = []

    for name, model in trained_models.items():
        report = evaluator.evaluate_model(
            model=model,
            X_test=X_test,
            y_test=test_labels,
            model_name=name,
            dataset_split_name="test"
        )
        test_reports.append(report)

    # Step 5: Model Comparison & Ranking
    logger.info("Step 5: Compiling Benchmark Comparison & Charts...")
    comparator = ModelComparison(results_dir="results")
    comp_df = comparator.generate_comparison_table(test_reports, primary_metric="f1_weighted")
    comparator.plot_comparison_charts(comp_df)
    best_model_info = comparator.get_best_model(comp_df)

    # Step 6: Chatbot Demonstration
    logger.info("Step 6: Verifying Chatbot Inference Engine...")
    chatbot = MentalHealthChatbotEngine()
    test_msg = "I've been feeling completely overwhelmed with panic and cannot concentrate at all."
    bot_out = chatbot.process_message(test_msg)

    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*80)
    print(comp_df.to_string(index=False))
    print("-" * 80)
    print(f"🏆 Best Model: {best_model_info.get('Model')}")
    print(f"   Accuracy:   {best_model_info.get('Accuracy')}")
    print(f"   F1-Score:   {best_model_info.get('F1_Score')}")
    print(f"   Confidence: {best_model_info.get('Confidence')}")
    print("="*80)

    print("\nCHATBOT VERIFICATION RESPONSE:")
    print(f"User Input: {test_msg}")
    print(f"Prediction: {bot_out['predicted_class']} (Confidence: {bot_out['confidence']:.2f})")
    print(f"Response:\n{bot_out['response']}")
    print("="*80 + "\n")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
