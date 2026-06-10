"""
Check dataset columns, split distribution, label distribution, and basic data quality.

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
    "is_code_mixed",
    "has_emoji",
    "has_hashtag",
    "contains_url",
    "text_length_chars",
    "token_count",
    "created_year",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file to check")
    args = parser.parse_args()

    path = Path(args.data)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATASET CHECK REPORT")
    print("=" * 60)

    print(f"\nFile: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns found:")
    for col in df.columns:
        print(f"- {col}")

    missing = [col for col in RECOMMENDED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in RECOMMENDED_COLUMNS]

    if missing:
        print("\nMissing recommended columns:")
        for col in missing:
            print(f"- {col}")
    else:
        print("\nAll recommended columns are present.")

    if extra:
        print("\nExtra columns found:")
        for col in extra:
            print(f"- {col}")

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
        empty_text = (df["text"].fillna("").astype(str).str.strip() == "").sum()
        print(f"Empty text values: {empty_text:,}")

    if "clean_text" in df.columns:
        empty_clean_text = (df["clean_text"].fillna("").astype(str).str.strip() == "").sum()
        print(f"Empty clean_text values: {empty_clean_text:,}")

    print("\nCheck completed.")


if __name__ == "__main__":
    main()