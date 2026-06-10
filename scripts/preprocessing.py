"""
Language-aware preprocessing script for the multilingual offensive-language dataset.

Usage:
    python scripts/preprocessing.py --input "full data.csv" --output "full_data_cleaned.csv"
"""

import argparse
import re
from pathlib import Path

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}", flags=re.UNICODE)


def normalize_repeated_characters(text: str) -> str:
    """
    Reduce very long repeated characters while keeping emphasis.
    Example: gooooood -> goood
    """
    return REPEATED_CHAR_RE.sub(r"\1\1\1", text)


def clean_text(text: str, language: str = "") -> str:
    """
    Conservative cleaning for multilingual offensive-language data.
    Removes platform noise while preserving offensive cues, hashtags,
    transliteration patterns, and Urdu/Pashto script information.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    language = str(language).lower().strip()

    # Remove URLs and user mentions.
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)

    # Keep hashtag text because hashtags may carry offensive/topic information.
    text = text.replace("#", " #")

    # Normalize repeated characters mainly for Roman Urdu and English.
    if language in {"roman urdu", "english"}:
        text = normalize_repeated_characters(text)

    # Preserve Arabic-script characters for Urdu and Pashto.
    text = SPACE_RE.sub(" ", text).strip()

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--text-column", default="text", help="Column containing original text")
    parser.add_argument("--language-column", default="language", help="Column containing language name")
    parser.add_argument("--clean-column", default="clean_text", help="Output cleaned text column")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.text_column not in df.columns:
        raise ValueError(f"Text column not found: {args.text_column}")

    if args.language_column not in df.columns:
        df[args.language_column] = ""

    df[args.clean_column] = [
        clean_text(text, lang)
        for text, lang in zip(df[args.text_column], df[args.language_column])
    ]

    before_rows = len(df)
    df = df[df[args.clean_column].astype(str).str.strip().str.len() > 0].copy()
    removed_rows = before_rows - len(df)

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved cleaned file: {output_path}")
    print(f"Rows saved: {len(df):,}")
    print(f"Empty rows removed: {removed_rows:,}")


if __name__ == "__main__":
    main()