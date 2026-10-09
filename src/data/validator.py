"""Dataset Validator Module for AI Chatbot Mental Health Project.

Performs schema checking, quality audits, missing value analysis, duplicate detection,
class balance checks, and data leakage detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DatasetValidator:
    """Validator class to check raw dataset quality, completeness, and integrity."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration dictionary."""
        self.config = config
        self.schema = config.get("schema", {})
        self.text_col = self.schema.get("text_column", "text")
        self.label_col = self.schema.get("label_column", "label")

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive validation on DataFrame.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Dict containing detailed validation metrics and status.
        """
        rows, cols = df.shape

        # Missing values analysis
        missing_series = df.isnull().sum()
        total_missing = int(missing_series.sum())
        missing_cols = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}

        # Duplicate analysis
        exact_row_dups = int(df.duplicated().sum())
        text_dups = int(df[self.text_col].duplicated().sum()) if self.text_col in df.columns else 0

        # Empty & short text records
        empty_text_records = 0
        short_text_records = 0
        if self.text_col in df.columns:
            text_series = df[self.text_col].astype(str)
            empty_mask = df[self.text_col].isna() | (text_series.str.strip() == "")
            empty_text_records = int(empty_mask.sum())
            word_counts = text_series.str.split().str.len()
            short_text_records = int((word_counts < 3).sum())

        # Class distribution and balance status
        class_dist = {}
        class_pcts = {}
        num_classes = 0
        balance_status = "Unknown"
        if self.label_col in df.columns:
            val_counts = df[self.label_col].value_counts()
            num_classes = int(len(val_counts))
            class_dist = {str(k): int(v) for k, v in val_counts.to_dict().items()}
            class_pcts = {str(k): float(round(v * 100, 2)) for k, v in df[self.label_col].value_counts(normalize=True).items()}
            
            # Balance determination
            max_pct = max(class_pcts.values()) if class_pcts else 0
            min_pct = min(class_pcts.values()) if class_pcts else 0
            if max_pct - min_pct <= 15:
                balance_status = "Balanced"
            elif max_pct - min_pct <= 35:
                balance_status = "Moderately Imbalanced"
            else:
                balance_status = "Highly Imbalanced"

        # Data Leakage Checks
        conflicting_label_texts = 0
        duplicate_post_ids = 0
        if self.text_col in df.columns and self.label_col in df.columns:
            dup_texts = df[df[self.text_col].duplicated(keep=False)]
            if len(dup_texts) > 0:
                grouped = dup_texts.groupby(self.text_col)[self.label_col].nunique()
                conflicting_label_texts = int((grouped > 1).sum())

        if "post_id" in df.columns:
            duplicate_post_ids = int(df["post_id"].duplicated().sum())

        leakage_detected = bool(conflicting_label_texts > 0 or text_dups > 0)

        # Build issues list
        issues: List[str] = []
        if total_missing > 0:
            issues.append(f"Missing values found in {len(missing_cols)} column(s).")
        if text_dups > 0:
            issues.append(f"Found {text_dups} duplicate text entries in dataset.")
        if conflicting_label_texts > 0:
            issues.append(f"Found {conflicting_label_texts} identical texts with conflicting target labels.")
        if empty_text_records > 0:
            issues.append(f"Found {empty_text_records} empty or whitespace-only text records.")

        status = "PASS" if len(issues) == 0 or (conflicting_label_texts == 0 and empty_text_records == 0) else "PARTIAL"

        report = {
            "status": status,
            "dataset_shape": {"rows": int(rows), "columns": int(cols)},
            "schema": {
                "text_column": self.text_col,
                "label_column": self.label_col,
                "columns_present": list(df.columns)
            },
            "missing_values": {
                "total_missing_cells": total_missing,
                "columns_with_missing": missing_cols
            },
            "duplicates": {
                "exact_row_duplicates": exact_row_dups,
                "text_duplicates": text_dups,
                "duplicate_percentage": float(round((exact_row_dups / rows) * 100, 2))
            },
            "text_quality": {
                "empty_text_records": empty_text_records,
                "short_text_records_lt_3_words": short_text_records
            },
            "target_analysis": {
                "label_column": self.label_col,
                "number_of_classes": num_classes,
                "class_distribution": class_dist,
                "class_percentages": class_pcts,
                "class_balance_status": balance_status
            },
            "data_leakage": {
                "identical_text_conflicting_labels": conflicting_label_texts,
                "duplicate_post_ids": duplicate_post_ids,
                "leakage_risk_detected": leakage_detected
            },
            "issues": issues
        }

        return report
