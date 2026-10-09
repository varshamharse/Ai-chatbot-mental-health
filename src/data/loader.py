"""Dataset Loader Module for AI Chatbot Mental Health Project.

Reads raw dataset files based on configuration settings without modifying raw files.
"""

import os
import yaml
import pandas as pd
from typing import Optional, Dict, Any


class DatasetLoader:
    """Configurable Dataset Loader supporting CSV, JSON, JSONL, Parquet, and Excel formats."""

    def __init__(self, config_path: str = "configs/dataset.yaml"):
        """Initialize DatasetLoader with configuration path."""
        self.config_path = config_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML dataset configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame.
        
        Args:
            file_path: Optional override for dataset file path.
            
        Returns:
            pd.DataFrame: Loaded immutable raw dataframe.
        """
        target_path = file_path or self.config.get("dataset", {}).get("path")
        if not target_path:
            raise ValueError("Dataset path is not specified in config or arguments.")

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Raw dataset file not found at path: {target_path}")

        file_format = self.config.get("dataset", {}).get("format", "").lower()
        if not file_format:
            file_format = os.path.splitext(target_path)[1].lstrip(".").lower()

        try:
            if file_format == "csv":
                df = pd.read_csv(target_path)
            elif file_format == "json":
                df = pd.read_json(target_path)
            elif file_format == "jsonl":
                df = pd.read_json(target_path, lines=True)
            elif file_format == "parquet":
                df = pd.read_parquet(target_path)
            elif file_format in ["xlsx", "xls"]:
                df = pd.read_excel(target_path)
            else:
                raise ValueError(f"Unsupported dataset file format: {file_format}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset file '{target_path}': {str(e)}")

        self._validate_schema(df)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that configured text and label columns exist in the loaded DataFrame."""
        schema = self.config.get("schema", {})
        text_col = schema.get("text_column")
        label_col = schema.get("label_column")

        missing_cols = []
        if text_col and text_col not in df.columns:
            missing_cols.append(text_col)
        if label_col and label_col not in df.columns:
            missing_cols.append(label_col)

        if missing_cols:
            raise KeyError(
                f"Configured column(s) {missing_cols} missing from loaded dataset. "
                f"Available columns: {list(df.columns[:10])}..."
            )


def load_raw_dataset(config_path: str = "configs/dataset.yaml") -> pd.DataFrame:
    """Helper function to load dataset quickly using config."""
    loader = DatasetLoader(config_path=config_path)
    return loader.load_dataset()
