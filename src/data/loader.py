"""Dataset Loader Module for AI Chatbot Mental Health Project.

Reads raw dataset files based on configuration settings without modifying raw files.
Supports automatic path resolution and format handling.
"""

import os
import yaml
import pandas as pd
from typing import Optional, Dict, Any, List


class DatasetLoader:
    """Configurable Dataset Loader supporting CSV, JSON, JSONL, Parquet, and Excel formats."""

    def __init__(self, config_path: str = "configs/dataset.yaml"):
        """Initialize DatasetLoader with configuration path."""
        self.config_path = self._resolve_path(config_path)
        self.config = self._load_config(self.config_path)

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to project root if not found directly."""
        if os.path.exists(path):
            return path
        # Check relative to repo root (one level up if inside scripts/src)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.join(base_dir, path)
        if os.path.exists(candidate):
            return candidate
        return path

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML dataset configuration file."""
        resolved = self._resolve_path(config_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _find_dataset_file(self, target_path: Optional[str]) -> str:
        """Find dataset file among configured path and fallbacks."""
        candidates = []
        if target_path:
            candidates.append(target_path)
        
        configured_path = self.config.get("dataset", {}).get("path")
        if configured_path and configured_path not in candidates:
            candidates.append(configured_path)

        fallbacks = self.config.get("dataset", {}).get("fallback_paths", [])
        for fb in fallbacks:
            if fb not in candidates:
                candidates.append(fb)

        # Standard repository dataset locations
        candidates.extend([
            "Dataset/Dreaddit_combine_data.csv",
            "data/raw/mental_health_dataset.csv",
            "data/raw/Dreaddit_combine_data.csv"
        ])

        for c in candidates:
            resolved = self._resolve_path(c)
            if os.path.exists(resolved):
                return resolved

        raise FileNotFoundError(f"Raw dataset file not found among candidates: {candidates}")

    def load_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame.
        
        Args:
            file_path: Optional override for dataset file path.
            
        Returns:
            pd.DataFrame: Loaded immutable raw dataframe.
        """
        resolved_path = self._find_dataset_file(file_path)
        file_format = self.config.get("dataset", {}).get("format", "").lower()
        if not file_format:
            file_format = os.path.splitext(resolved_path)[1].lstrip(".").lower()

        try:
            if file_format == "csv":
                df = pd.read_csv(resolved_path)
            elif file_format == "json":
                df = pd.read_json(resolved_path)
            elif file_format == "jsonl":
                df = pd.read_json(resolved_path, lines=True)
            elif file_format == "parquet":
                df = pd.read_parquet(resolved_path)
            elif file_format in ["xlsx", "xls"]:
                df = pd.read_excel(resolved_path)
            else:
                raise ValueError(f"Unsupported dataset file format: {file_format}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset file '{resolved_path}': {str(e)}")

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
