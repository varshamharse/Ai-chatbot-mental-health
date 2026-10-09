"""Unit tests for dataset loading, validation, and splitting."""

import pytest
import pandas as pd
import numpy as np
from src.data.splitter import DatasetSplitter
from src.data.dataset import MentalHealthDataset


def test_dataset_splitter():
    df = pd.DataFrame({
        "text": [f"sample text {i}" for i in range(100)],
        "label": [0] * 50 + [1] * 50
    })
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    train_df, val_df, test_df = splitter.split(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    # Check stratification
    assert (train_df["label"] == 1).sum() == 35


def test_mental_health_dataset_container():
    df = pd.DataFrame({
        "text": ["feeling good", "feeling stressed"],
        "label": [0, 1],
        "confidence": [0.8, 1.0]
    })
    dataset = MentalHealthDataset(df)
    assert len(dataset) == 2
    assert dataset.texts == ["feeling good", "feeling stressed"]
    assert list(dataset.labels) == [0, 1]
    assert list(dataset.confidences) == [0.8, 1.0]
