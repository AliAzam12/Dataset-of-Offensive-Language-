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
    Example: "gooooood" -> "goood"
    """
    return REPEATED_CHAR_RE.sub(r"\1\1\1", text)


def clean_text(text: str, language: str = "") -> str:
    """
    Conservative cleaning for multilingual offensive-language data.
    It removes platform noise while preserving useful offensive cues.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    language = str(language).lower().strip()

    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)

    # Keep hashtags as words because hashtags may carry target/topic information.
    text = text.replace("#", " #")

    # Conservative repeated-character normalisation, especially useful for Roman Urdu/English.
    if language in {"roman urdu", "english", "en", "ru"}:
        text = normalize_repeated_characters(text)

    # Do not remove Arabic-script characters because Urdu/Pashto cues may be lost.
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

    df = pd.read_csv(input_path)

    if args.text_column not in df.columns:
        raise ValueError(f"Text column not found: {args.text_column}")

    if args.language_column not in df.columns:
        df[args.language_column] = ""

    df[args.clean_column] = [
        clean_text(text, lang)
        for text, lang in zip(df[args.text_column], df[args.language_column])
    ]

    df = df[df[args.clean_column].astype(str).str.len() > 0].copy()
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved cleaned file: {output_path}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
