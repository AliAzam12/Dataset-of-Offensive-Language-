"""
Create language/label summary counts for the dataset.

Usage:
    python scripts/create_summary_counts.py --data "full data.csv" --output "dataset_summary_counts_generated.csv"
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="dataset_summary_counts_generated.csv")
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--label-column", default="final_label")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.language_column not in df.columns:
        raise ValueError(f"Language column not found: {args.language_column}")

    if args.label_column not in df.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    summary = (
        df.groupby([args.language_column, args.label_column])
        .size()
        .reset_index(name="count")
        .sort_values([args.language_column, args.label_column])
    )

    pivot = (
        summary.pivot(index=args.language_column, columns=args.label_column, values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    pivot["total"] = pivot.drop(columns=[args.language_column]).sum(axis=1)
    pivot.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved summary counts to: {args.output}")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
