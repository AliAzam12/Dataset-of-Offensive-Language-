"""
Check dataset columns, split distribution, and label distribution.

Usage:
    python scripts/check_dataset.py --data "full data.csv"
"""

import argparse
from pathlib import Path

import pandas as pd


RECOMMENDED_COLUMNS = [
    "id",
    "text",
    "clean_text",
    "language",
    "script",
    "source_platform",
    "source_dataset",
    "original_label",
    "final_label",
    "label_id",
    "split",
    "topic",
    "metadata_flags",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file to check")
    args = parser.parse_args()

    path = Path(args.data)
    df = pd.read_csv(path)

    print(f"File: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns found:")
    for col in df.columns:
        print(f"- {col}")

    missing = [col for col in RECOMMENDED_COLUMNS if col not in df.columns]
    if missing:
        print("\nRecommended columns missing:")
        for col in missing:
            print(f"- {col}")
    else:
        print("\nAll recommended columns are present.")

    if "language" in df.columns:
        print("\nLanguage distribution:")
        print(df["language"].value_counts(dropna=False).to_string())

    if "final_label" in df.columns:
        print("\nLabel distribution:")
        print(df["final_label"].value_counts(dropna=False).to_string())

    if "split" in df.columns:
        print("\nSplit distribution:")
        print(df["split"].value_counts(dropna=False).to_string())

    if "id" in df.columns:
        duplicates = df["id"].duplicated().sum()
        print(f"\nDuplicate IDs: {duplicates:,}")

    if "text" in df.columns:
        empty_text = df["text"].isna().sum() + (df["text"].fillna("").astype(str).str.strip() == "").sum()
        print(f"Empty text values: {empty_text:,}")


if __name__ == "__main__":
    main()
