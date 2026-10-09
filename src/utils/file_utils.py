"""File I/O helpers for loading/saving configurations, data, and serialized models."""

import os
import json
import pickle
from typing import Any, Dict
import yaml
import pandas as pd


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def save_model_artifact(artifact: Any, file_path: str) -> None:
    """Serialize model or preprocessor to disk using pickle."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(artifact, f)


def load_model_artifact(file_path: str) -> Any:
    """Deserialize model or preprocessor artifact from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact file not found: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)
