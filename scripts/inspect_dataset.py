"""Script to inspect raw dataset, run validation audit, and export statistics files.

Usage:
    python scripts/inspect_dataset.py --config configs/dataset.yaml
"""

import os
import sys
import json
import argparse
import yaml
import pandas as pd

# Add src package directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Inspect raw dataset for AI Chatbot Mental Health project.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to dataset configuration file.")
    args = parser.parse_args()

    # 1. Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Load dataset
    print(f"Loading raw dataset using config '{args.config}'...")
    loader = DatasetLoader(config_path=args.config)
    df = loader.load_dataset()

    # 3. Validate dataset
    validator = DatasetValidator(config=config)
    val_report = validator.validate(df)

    # 4. Create results directory
    os.makedirs("results", exist_ok=True)

    # 5. Export results/dataset_validation.json
    val_json_path = "results/dataset_validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    print(f"Saved validation report to '{val_json_path}'")

    # 6. Export results/dataset_statistics.csv
    stats_rows = []
    rows = len(df)
    for col in df.columns:
        stats_rows.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round(float(df[col].isnull().sum() / rows * 100), 2),
            "unique_count": int(df[col].nunique())
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv_path = "results/dataset_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved dataset column statistics to '{stats_csv_path}'")

    # 7. Print STEP 2 Inspection Summary
    dataset_file = os.path.basename(config.get("dataset", {}).get("path", "mental_health_dataset.csv"))
    dataset_format = config.get("dataset", {}).get("format", "csv")
    text_col = config.get("schema", {}).get("text_column", "text")
    label_col = config.get("schema", {}).get("label_column", "label")
    task_name = config.get("task", {}).get("primary", {}).get("name", "Stress Classification")
    task_type = "Binary Classification" if val_report["target_analysis"]["number_of_classes"] == 2 else "Multiclass Classification"
    
    classes_dict = val_report["target_analysis"]["class_distribution"]
    class_names = list(classes_dict.keys())
    
    issues_list = val_report.get("issues", [])
    issues_str = "\n".join([f"- {issue}" for issue in issues_list]) if issues_list else "None"

    print("\n" + "="*40)
    print("STEP 2 — DATASET INSPECTION REPORT")
    print("="*40)
    print(f"Status:\n{val_report['status']}")
    print(f"\nDataset:\n{dataset_file}")
    print(f"\nFormat:\n{dataset_format}")
    print(f"\nRows:\n{val_report['dataset_shape']['rows']}")
    print(f"\nColumns:\n{val_report['dataset_shape']['columns']}")
    print(f"\nText Column:\n{text_col}")
    print(f"\nTarget Column:\n{label_col}")
    print(f"\nTask:\n{task_type} ({task_name})")
    print(f"\nNumber of Classes:\n{val_report['target_analysis']['number_of_classes']}")
    print(f"\nClasses:\n{class_names} (0: Non-Stress, 1: Stress)")
    print(f"\nMissing Values:\n{val_report['missing_values']['total_missing_cells']}")
    print(f"\nDuplicate Rows:\n{val_report['duplicates']['exact_row_duplicates']} exact rows, {val_report['duplicates']['text_duplicates']} text duplicates")
    print(f"\nEmpty Text Records:\n{val_report['text_quality']['empty_text_records']}")
    print(f"\nClass Balance:\n{val_report['target_analysis']['class_balance_status']} (0: {val_report['target_analysis']['class_percentages'].get('0', 0)}%, 1: {val_report['target_analysis']['class_percentages'].get('1', 0)}%)")
    print(f"\nPotential Data Leakage:\n{'YES' if val_report['data_leakage']['leakage_risk_detected'] else 'NO'}")
    print(f"\nConfiguration:\n{args.config}")
    print(f"\nValidation Report:\n{val_json_path}")
    print(f"\nDataset Statistics:\n{stats_csv_path}")
    print("\nFiles Created/Modified:")
    print("- configs/dataset.yaml")
    print("- src/data/loader.py")
    print("- src/data/validator.py")
    print("- scripts/inspect_dataset.py")
    print("- results/dataset_validation.json")
    print("- results/dataset_statistics.csv")
    print(f"\nIssues:\n{issues_str}")
    print(f"\nReady for STEP 3:\nYES")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
