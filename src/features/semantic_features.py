"""Semantic and lexical meta-features extraction module."""

import numpy as np
import pandas as pd
from typing import List


class SemanticFeatureExtractor:
    """Extracts meta-linguistic features such as length, punctuation ratios, and sentiment cues."""

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract dense linguistic feature matrix from texts.
        
        Args:
            texts: List of input strings.
            
        Returns:
            np.ndarray of shape (n_samples, n_features)
        """
        features = []
        for text in texts:
            if not text:
                features.append([0, 0, 0, 0, 0, 0])
                continue

            char_len = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_len = (char_len / word_count) if word_count > 0 else 0
            qmark_count = text.count("?")
            exclam_count = text.count("!")
            uppercase_count = sum(1 for c in text if c.isupper())
            uppercase_ratio = (uppercase_count / char_len) if char_len > 0 else 0

            features.append([
                char_len,
                word_count,
                avg_word_len,
                qmark_count,
                exclam_count,
                uppercase_ratio
            ])

        return np.array(features, dtype=float)

    @property
    def feature_names(self) -> List[str]:
        return [
            "char_length",
            "word_count",
            "avg_word_length",
            "question_marks",
            "exclamation_marks",
            "uppercase_ratio"
        ]
