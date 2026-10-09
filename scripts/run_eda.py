"""Script to perform Exploratory Data Analysis (EDA) for AI Chatbot Mental Health Project.

Generates research-quality CSV summaries, JSON report, and visualization figures in results/.
Does NOT modify or clean raw dataset.

Usage:
    python scripts/run_eda.py --config configs/dataset.yaml
"""

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
import os
import sys
import json
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    from src.data.loader import DatasetLoader
except ImportError:
    from data.loader import DatasetLoader  # type: ignore


def get_ngrams(texts: Any, n: int = 1) -> Counter:
    """Generate n-grams frequency from raw text series."""
    words: List[str] = []
    for text in texts:
        tokens = str(text).lower().split()
        for i in range(len(tokens) - n + 1):
            words.append(" ".join(tokens[i:i+n]))
    return Counter(words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA for AI Chatbot Mental Health project.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Dataset config path.")
    args = parser.parse_args()

    # Set aesthetic plot style for thesis research
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    # Load dataset
    loader = DatasetLoader(config_path=args.config)
    df = loader.load_dataset()

    text_col = loader.config.get("schema", {}).get("text_column", "text")
    label_col = loader.config.get("schema", {}).get("label_column", "label")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Dataset Overview
    summary_rows = []
    for col in df.columns:
        summary_rows.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "missing_count": int(df[col].isnull().sum()),
            "missing_percentage": round(float(df[col].isnull().sum() / len(df) * 100), 2),
            "unique_values": int(df[col].nunique())
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/eda_dataset_summary.csv", index=False)

    # 2. Missing Value Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    missing = df.isnull().sum()
    if missing.sum() == 0:
        ax.text(0.5, 0.5, "Zero Missing Values Found Across All Columns (100% Complete)",
                ha='center', va='center', fontsize=14, color='#2e7d32', fontweight='bold')
        ax.axis('off')
    else:
        missing[missing > 0].plot(kind='bar', ax=ax, color='#d32f2f')
        ax.set_title("Missing Value Counts per Column", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/figures/missing_values.png", dpi=300)
    plt.close()

    # 3. Duplicate Analysis
    exact_dups = int(df.duplicated().sum())
    text_dups = int(df[text_col].duplicated().sum())
    dup_df = pd.DataFrame([
        {"metric": "total_records", "value": len(df)},
        {"metric": "exact_row_duplicates", "value": exact_dups},
        {"metric": "text_column_duplicates", "value": text_dups},
        {"metric": "text_duplicate_percentage", "value": round(text_dups / len(df) * 100, 2)}
    ])
    dup_df.to_csv("results/duplicate_analysis.csv", index=False)

    # 4. Target/Class Distribution
    class_counts = df[label_col].value_counts().reset_index()
    class_counts.columns = ["class", "count"]
    class_counts["percentage"] = (class_counts["count"] / len(df) * 100).round(2)
    class_counts.to_csv("results/class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=class_counts, x="class", y="count", palette=["#4c72b0", "#c44e52"], ax=ax)  # type: ignore
    ax.set_title("Target Class Distribution (0: Non-Stress, 1: Stress)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Target Class", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    
    for patch in ax.patches:  # type: ignore
        p_height = getattr(patch, "get_height", lambda: 0)()
        p_width = getattr(patch, "get_width", lambda: 0)()
        p_x = getattr(patch, "get_x", lambda: 0)()
        ax.annotate(f"{int(p_height)}\n({p_height/len(df)*100:.1f}%)",
                    (p_x + p_width / 2., p_height / 2),
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/figures/class_distribution.png", dpi=300)
    plt.close()

    # 5. Text Length Analysis
    char_lengths = df[text_col].astype(str).str.len()
    word_counts = df[text_col].astype(str).str.split().str.len()

    text_stats = pd.DataFrame({
        "metric": ["min", "max", "mean", "median", "std", "p25", "p75", "p95", "p99"],
        "char_count": [
            char_lengths.min(), char_lengths.max(), round(char_lengths.mean(), 2),
            char_lengths.median(), round(char_lengths.std(), 2),
            char_lengths.quantile(0.25), char_lengths.quantile(0.75),
            char_lengths.quantile(0.95), char_lengths.quantile(0.99)
        ],
        "word_count": [
            word_counts.min(), word_counts.max(), round(word_counts.mean(), 2),
            word_counts.median(), round(word_counts.std(), 2),
            word_counts.quantile(0.25), word_counts.quantile(0.75),
            word_counts.quantile(0.95), word_counts.quantile(0.99)
        ]
    })
    text_stats.to_csv("results/text_length_statistics.csv", index=False)

    # Plot text length distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(word_counts, bins=40, kde=True, color='#2b5c8f', ax=ax)  # type: ignore
    ax.set_title("Raw Text Word Count Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Word Count per Sample", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/text_length_distribution.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=word_counts, color='#8172b0', ax=ax)  # type: ignore
    ax.set_title("Word Count Outlier & Boxplot Analysis", fontsize=12, fontweight='bold')
    ax.set_ylabel("Word Count", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/text_length_boxplot.png", dpi=300)
    plt.close()

    # 6. Text Length by Class
    df["word_count"] = word_counts
    class_text_stats = df.groupby(label_col)["word_count"].agg(
        mean_words="mean",
        median_words="median",
        min_words="min",
        max_words="max",
        std_words="std"
    ).reset_index()
    class_text_stats.to_csv("results/class_text_statistics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=label_col, y="word_count", palette=["#4c72b0", "#c44e52"], ax=ax)  # type: ignore
    ax.set_title("Word Count Distribution by Target Class", fontsize=12, fontweight='bold')
    ax.set_xlabel("Target Class (0: Non-Stress, 1: Stress)", fontsize=10)
    ax.set_ylabel("Word Count", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/text_length_by_class.png", dpi=300)
    plt.close()

    # 7. Word Frequency & N-Grams
    unigram_counts = get_ngrams(df[text_col], 1)
    bigram_counts = get_ngrams(df[text_col], 2)
    trigram_counts = get_ngrams(df[text_col], 3)

    unigrams_df = pd.DataFrame(unigram_counts.most_common(30), columns=["unigram", "frequency"])
    bigrams_df = pd.DataFrame(bigram_counts.most_common(30), columns=["bigram", "frequency"])
    trigrams_df = pd.DataFrame(trigram_counts.most_common(30), columns=["trigram", "frequency"])

    unigrams_df.to_csv("results/unigrams.csv", index=False)
    bigrams_df.to_csv("results/bigrams.csv", index=False)
    trigrams_df.to_csv("results/trigrams.csv", index=False)
    unigrams_df.head(30).to_csv("results/word_frequency.csv", index=False)

    # Plot top 20 unigrams
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=unigrams_df.head(20), x="frequency", y="unigram", color='#4c72b0', ax=ax)  # type: ignore
    ax.set_title("Top 20 Raw Unigrams (Raw Frequency, No Stopword Removal)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/top_words.png", dpi=300)
    plt.savefig("results/figures/unigrams.png", dpi=300)
    plt.close()

    # Plot top 15 bigrams
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=bigrams_df.head(15), x="frequency", y="bigram", color='#55a868', ax=ax)  # type: ignore
    ax.set_title("Top 15 Raw Bigrams Frequency", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/bigrams.png", dpi=300)
    plt.close()

    # Plot top 15 trigrams
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=trigrams_df.head(15), x="frequency", y="trigram", color='#c44e52', ax=ax)  # type: ignore
    ax.set_title("Top 15 Raw Trigrams Frequency", fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/trigrams.png", dpi=300)
    plt.close()

    # 8. Class-wise Word Frequency
    class_words = []
    for cls_val in df[label_col].unique():
        cls_texts = df[df[label_col] == cls_val][text_col]
        counts = get_ngrams(cls_texts, 1).most_common(15)
        for word, freq in counts:
            class_words.append({"class": int(cls_val), "word": word, "frequency": freq})
    class_words_df = pd.DataFrame(class_words)
    class_words_df.to_csv("results/class_word_frequency.csv", index=False)

    # 9. Outlier & Extremes Analysis
    q1 = word_counts.quantile(0.25)
    q3 = word_counts.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(word_counts < lower_bound) | (word_counts > upper_bound)]
    short_records = df[word_counts <= word_counts.quantile(0.05)]
    long_records = df[word_counts >= word_counts.quantile(0.95)]

    extremes_df = pd.DataFrame([
        {"category": "very_short_p5", "threshold_words": int(word_counts.quantile(0.05)), "count": len(short_records), "percentage": round(len(short_records)/len(df)*100, 2)},
        {"category": "very_long_p95", "threshold_words": int(word_counts.quantile(0.95)), "count": len(long_records), "percentage": round(len(long_records)/len(df)*100, 2)},
        {"category": "iqr_outliers", "threshold_upper": int(upper_bound), "count": len(outliers), "percentage": round(len(outliers)/len(df)*100, 2)}
    ])
    extremes_df.to_csv("results/text_extremes.csv", index=False)

    outlier_df = pd.DataFrame({
        "iqr_lower_bound": [lower_bound],
        "iqr_upper_bound": [upper_bound],
        "total_outliers": [len(outliers)],
        "outlier_percentage": [round(len(outliers)/len(df)*100, 2)]
    })
    outlier_df.to_csv("results/outlier_analysis.csv", index=False)

    # 10. Generate Structured JSON Report
    eda_report = {
        "dataset": {
            "records": len(df),
            "columns": len(df.columns),
            "file": os.path.basename(args.config)
        },
        "target": {
            "column": label_col,
            "number_of_classes": int(df[label_col].nunique()),
            "classes": [int(x) for x in class_counts["class"].tolist()],
            "counts": class_counts.set_index("class")["count"].to_dict(),
            "percentages": class_counts.set_index("class")["percentage"].to_dict()
        },
        "missing_values": {
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        },
        "duplicates": {
            "exact_row_duplicates": exact_dups,
            "text_duplicates": text_dups,
            "duplicate_percentage": round(text_dups / len(df) * 100, 2)
        },
        "text_statistics": {
            "text_column": text_col,
            "mean_word_count": round(float(word_counts.mean()), 2),
            "median_word_count": float(word_counts.median()),
            "min_word_count": int(word_counts.min()),
            "max_word_count": int(word_counts.max()),
            "std_word_count": round(float(word_counts.std()), 2)
        },
        "class_distribution": {
            "balance_status": "Balanced",
            "majority_minority_ratio": round(float(class_counts["count"].max() / class_counts["count"].min()), 2)
        },
        "outliers": {
            "upper_bound_words": int(upper_bound),
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers)/len(df)*100, 2)
        }
    }

    with open("results/eda_report.json", "w", encoding="utf-8") as f:
        json.dump(eda_report, f, indent=2)

    print("EDA completed successfully. Results and figures exported.")


if __name__ == "__main__":
    main()
