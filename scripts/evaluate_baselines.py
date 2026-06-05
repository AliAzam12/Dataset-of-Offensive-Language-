"""
Evaluate simple TF-IDF baselines on the official train/validation/test split.

Usage:
    python scripts/evaluate_baselines.py \
      --train "train data.csv" \
      --valid "validation data.csv" \
      --test "test data.csv" \
      --text-column clean_text \
      --label-column final_label
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def load_split(path: str, text_column: str, label_column: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)

    if text_column not in df.columns:
        # Fallback to original text if clean_text is not present.
        if text_column == "clean_text" and "text" in df.columns:
            text_column = "text"
        else:
            raise ValueError(f"Text column not found in {path}: {text_column}")

    if label_column not in df.columns:
        raise ValueError(f"Label column not found in {path}: {label_column}")

    x = df[text_column].fillna("").astype(str)
    y = df[label_column].astype(str)
    return x, y


def build_features() -> FeatureUnion:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    return FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    pipe = Pipeline([
        ("features", build_features()),
        ("classifier", model),
    ])

    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(pd.Series(result).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--text-column", default="clean_text")
    parser.add_argument("--label-column", default="final_label")
    parser.add_argument("--output", default="baseline_results.csv")
    args = parser.parse_args()

    x_train, y_train = load_split(args.train, args.text_column, args.label_column)
    x_valid, y_valid = load_split(args.valid, args.text_column, args.label_column)
    x_test, y_test = load_split(args.test, args.text_column, args.label_column)

    # Combine train + validation for final baseline training after model selection.
    x_train_full = pd.concat([x_train, x_valid], ignore_index=True)
    y_train_full = pd.concat([y_train, y_valid], ignore_index=True)

    models = [
        ("Naive Bayes", MultinomialNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Linear SVM", LinearSVC(class_weight="balanced")),
    ]

    results = []
    for name, model in models:
        results.append(evaluate_model(name, model, x_train_full, y_train_full, x_test, y_test))

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
