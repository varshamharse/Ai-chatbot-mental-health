"""High-level Trainer orchestration interface."""

from typing import Dict, Any, List, Optional
import numpy as np

from .train_baseline import BaselineTrainer
from src.utils.logger import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    """Unified high-level trainer class."""

    def __init__(self, config_path: str = "configs/models.yaml"):
        self.config_path = config_path
        self.baseline_trainer = BaselineTrainer(config_path=config_path)

    def train_baseline_models(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train all baseline models and evaluate on validation set."""
        return self.baseline_trainer.train_all(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels
        )
