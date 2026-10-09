"""Dataset abstraction class for Mental Health AI Chatbot."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd


class MentalHealthDataset:
    """Wrapper for handling mental health text, labels, and annotator confidence."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "label",
        confidence_column: Optional[str] = "confidence",
        id_column: Optional[str] = "post_id"
    ):
        """Initialize MentalHealthDataset.
        
        Args:
            df: Underlying pandas DataFrame.
            text_column: Column name containing raw or preprocessed text.
            label_column: Column name containing target label.
            confidence_column: Optional column for annotator confidence.
            id_column: Optional column for post ID.
        """
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.confidence_column = confidence_column
        self.id_column = id_column

    def __len__(self) -> int:
        return len(self.df)

    @property
    def texts(self) -> List[str]:
        """Return list of text samples."""
        return self.df[self.text_column].astype(str).tolist()

    @property
    def labels(self) -> np.ndarray:
        """Return numpy array of target labels."""
        return self.df[self.label_column].to_numpy()

    @property
    def confidences(self) -> Optional[np.ndarray]:
        """Return annotator confidence scores if present."""
        if self.confidence_column and self.confidence_column in self.df.columns:
            return self.df[self.confidence_column].to_numpy()
        return None

    def get_class_counts(self) -> dict:
        """Get class distribution counts."""
        return self.df[self.label_column].value_counts().to_dict()

    def to_dataframe(self) -> pd.DataFrame:
        """Return underlying DataFrame."""
        return self.df
