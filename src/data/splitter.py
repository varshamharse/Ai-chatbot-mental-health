"""Dataset Splitter Module for AI Chatbot Mental Health Project.

Performs stratified train, validation, and test splits with fixed random seeds.
Guarantees zero data leakage and preserves class distributions across splits.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Handles reproducible train/validation/test dataset splitting."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        stratify_column: Optional[str] = "label",
    ):
        """Initialize DatasetSplitter.
        
        Args:
            train_ratio: Fraction of dataset for training (default: 0.70).
            val_ratio: Fraction of dataset for validation (default: 0.15).
            test_ratio: Fraction of dataset for testing (default: 0.15).
            random_seed: Seed for reproducibility (default: 42).
            stratify_column: Column name to stratify on (default: 'label').
        """
        total = round(train_ratio + val_ratio + test_ratio, 4)
        if total != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify_column = stratify_column

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DatasetSplitter":
        """Instantiate splitter from dataset configuration dictionary."""
        split_cfg = config.get("split", {})
        schema_cfg = config.get("schema", {})
        stratify_col = schema_cfg.get("label_column", "label") if split_cfg.get("stratify", True) else None

        return cls(
            train_ratio=float(split_cfg.get("train", 0.70)),
            val_ratio=float(split_cfg.get("validation", 0.15)),
            test_ratio=float(split_cfg.get("test", 0.15)),
            random_seed=int(split_cfg.get("random_seed", 42)),
            stratify_column=stratify_col,
        )

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test subsets.
        
        Args:
            df: Cleaned dataframe.
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        stratify_target = df[self.stratify_column] if (self.stratify_column and self.stratify_column in df.columns) else None

        # First split: train vs (val + test)
        temp_ratio = self.val_ratio + self.test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=self.random_seed,
            stratify=stratify_target,
            shuffle=True
        )

        # Second split: val vs test from temp_df
        val_relative_ratio = self.val_ratio / temp_ratio
        temp_stratify = temp_df[self.stratify_column] if (self.stratify_column and self.stratify_column in temp_df.columns) else None

        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_relative_ratio,
            random_state=self.random_seed,
            stratify=temp_stratify,
            shuffle=True
        )

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = "data/processed"
    ) -> Dict[str, str]:
        """Save partitioned dataframes to CSV files.
        
        Args:
            train_df: Training set.
            val_df: Validation set.
            test_df: Test set.
            output_dir: Target output directory.
            
        Returns:
            Dictionary of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {
            "train": os.path.join(output_dir, "train.csv"),
            "val": os.path.join(output_dir, "val.csv"),
            "test": os.path.join(output_dir, "test.csv")
        }

        train_df.to_csv(paths["train"], index=False)
        val_df.to_csv(paths["val"], index=False)
        test_df.to_csv(paths["test"], index=False)

        return paths
