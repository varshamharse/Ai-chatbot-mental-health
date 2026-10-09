"""Baseline model training pipeline for all 5 classical ML models."""

import os
import time
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.features.tfidf import TFIDFFeatureExtractor
from src.models.baseline import (
    LogisticRegressionModel,
    NaiveBayesModel,
    SVMModel,
    RandomForestModel,
    KNNModel
)
from src.evaluation.evaluator import ModelEvaluator
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.file_utils import load_yaml, save_json

logger = get_logger("baseline_training")


class BaselineTrainer:
    """Orchestrates training and validation for Random Forest, SVM, Logistic Regression, KNN, and Naive Bayes."""

    def __init__(
        self,
        config_path: str = "configs/models.yaml",
        vectorizer_path: str = "models/vectorizers/tfidf_vectorizer.pkl",
        checkpoints_dir: str = "models/checkpoints",
        best_model_dir: str = "models/best_model"
    ):
        self.config_path = config_path
        self.vectorizer_path = vectorizer_path
        self.checkpoints_dir = checkpoints_dir
        self.best_model_dir = best_model_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)

        self.models_config = self._load_config()
        self.evaluator = ModelEvaluator()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            return load_yaml(self.config_path)
        return {}

    def get_model_instances(self) -> Dict[str, Any]:
        """Instantiate all 5 baseline models from configuration."""
        baseline_cfg = self.models_config.get("baseline", {})

        models = {
            "logistic_regression": LogisticRegressionModel(
                **baseline_cfg.get("logistic_regression", {"C": 1.0, "max_iter": 1000, "random_state": 42})
            ),
            "svm": SVMModel(
                **baseline_cfg.get("svm", {"C": 1.0, "kernel": "linear", "probability": True, "random_state": 42})
            ),
            "random_forest": RandomForestModel(
                **baseline_cfg.get("random_forest", {"n_estimators": 100, "random_state": 42, "n_jobs": -1})
            ),
            "knn": KNNModel(
                **baseline_cfg.get("knn", {"n_neighbors": 5, "weights": "distance", "metric": "cosine", "n_jobs": -1})
            ),
            "naive_bayes": NaiveBayesModel(
                **baseline_cfg.get("naive_bayes", {"alpha": 1.0})
            )
        }
        return models

    def train_all(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Fit feature extractor on train data, train all 5 models, and evaluate on validation set.
        
        Args:
            train_texts: Preprocessed training text samples.
            train_labels: Ground truth training labels.
            val_texts: Preprocessed validation text samples.
            val_labels: Validation ground truth labels.
            
        Returns:
            Dictionary containing trained models, validation metrics, and best model metadata.
        """
        set_seed(42)
        logger.info(f"Fitting TF-IDF vectorizer on {len(train_texts)} training samples...")
        
        feat_cfg = self.models_config.get("feature_extraction", {})
        vectorizer = TFIDFFeatureExtractor(
            max_features=feat_cfg.get("max_features", 5000),
            ngram_range=tuple(feat_cfg.get("ngram_range", [1, 2])),
            sublinear_tf=feat_cfg.get("sublinear_tf", True),
            min_df=feat_cfg.get("min_df", 2),
            max_df=feat_cfg.get("max_df", 0.95)
        )
        X_train = vectorizer.fit_transform(train_texts)
        vectorizer.save(self.vectorizer_path)
        logger.info(f"Saved TF-IDF vectorizer to {self.vectorizer_path}")

        X_val = vectorizer.transform(val_texts) if val_texts is not None else None

        models = self.get_model_instances()
        trained_models = {}
        val_reports = []
        best_val_f1 = -1.0
        best_model_name = None

        for name, model in models.items():
            logger.info(f"Training model: {name.replace('_', ' ').title()}...")
            t0 = time.perf_counter()
            model.fit(X_train, train_labels)
            train_time = time.perf_counter() - t0
            trained_models[name] = model

            # Save checkpoint
            ckpt_path = os.path.join(self.checkpoints_dir, f"{name}.pkl")
            model.save(ckpt_path)
            logger.info(f"Model {name} trained in {train_time:.2f}s and saved to {ckpt_path}")

            # Validate
            if X_val is not None and val_labels is not None:
                report = self.evaluator.evaluate_model(
                    model=model,
                    X_test=X_val,
                    y_test=val_labels,
                    model_name=name,
                    training_time_sec=train_time,
                    dataset_split_name="validation"
                )
                val_reports.append(report)
                val_f1 = report["metrics"]["f1_weighted"]
                logger.info(f"Validation F1-score for {name}: {val_f1:.4f} (Accuracy: {report['metrics']['accuracy']:.4f})")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_name = name

        # Save best model
        if best_model_name:
            logger.info(f"Designating Best Model: {best_model_name} (Val F1: {best_val_f1:.4f})")
            best_model = trained_models[best_model_name]
            best_model_path = os.path.join(self.best_model_dir, "best_model.pkl")
            best_model.save(best_model_path)

            meta = {
                "best_model_name": best_model_name,
                "validation_f1_weighted": best_val_f1,
                "vectorizer_path": self.vectorizer_path,
                "model_path": best_model_path,
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_json(meta, os.path.join(self.best_model_dir, "metadata.json"))

        return {
            "trained_models": trained_models,
            "validation_reports": val_reports,
            "best_model_name": best_model_name,
            "vectorizer": vectorizer
        }
