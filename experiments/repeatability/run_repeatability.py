"""
Phase 8: Repeatability Experiment.
Evaluates representative models across 5 independent random seeds: 42, 123, 456, 789, 2026.
Dataset splits remain strictly fixed.
Computes:
- Individual Macro-F1 values
- Mean
- Standard deviation (sample std, ddof=1)
- Relative standard deviation (CV % = std / mean * 100)
- Min and Max
- 95% t-based confidence interval
Saves:
- results/repeatability_runs.csv
- results/repeatability_summary.csv
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.evaluation.metrics import compute_metrics
from src.models.deep_learning_models import Vocab, TextDataset, TextCNN_BiLSTM, train_single_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEEDS = [42, 123, 456, 789, 2026]


def run_repeatability_study():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "validation.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))

    raw_runs = []

    # 1. Linear SVM across 5 seeds
    logging.info("Starting Repeatability Study for Linear SVM...")
    for seed in SEEDS:
        seed_everything(seed)
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        X_train = vec.fit_transform(train_df["clean_text"])
        X_test = vec.transform(test_df["clean_text"])

        t0 = time.time()
        clf = LinearSVC(C=0.1, max_iter=2000, random_state=seed)
        clf.fit(X_train, train_df["label_id"])
        train_time = round(time.time() - t0, 3)

        preds = clf.predict(X_test)
        metrics = compute_metrics(test_df["label_id"].values, preds)

        raw_runs.append({
            "model": "Linear_SVM",
            "family": "Traditional ML",
            "seed": seed,
            "train_time_sec": train_time,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4)
        })
        logging.info(f"  [Linear SVM] Seed {seed}: Macro-F1 = {metrics['macro_f1']:.4f}")

    # 2. CNN-BiLSTM across 5 seeds
    logging.info("Starting Repeatability Study for CNN-BiLSTM...")
    vocab = Vocab(max_size=5000)
    vocab.build_vocab(train_df["clean_text"])
    vocab_size = len(vocab.w2i)

    train_ds = TextDataset(train_df["clean_text"].tolist(), train_df["label_id"].tolist(), vocab, max_len=64)
    val_ds = TextDataset(val_df["clean_text"].tolist(), val_df["label_id"].tolist(), vocab, max_len=64)
    test_ds = TextDataset(test_df["clean_text"].tolist(), test_df["label_id"].tolist(), vocab, max_len=64)

    val_y = val_df["label_id"].values
    val_langs = val_df["language"].values
    test_y = test_df["label_id"].values
    test_langs = test_df["language"].values

    for seed in SEEDS:
        seed_everything(seed)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        res = train_single_model(
            TextCNN_BiLSTM, f"CNN-BiLSTM_seed{seed}",
            train_loader, val_loader, test_loader,
            val_y, val_langs, test_y, test_langs,
            vocab_size=vocab_size, max_epochs=10, patience=3
        )

        raw_runs.append({
            "model": "CNN-BiLSTM",
            "family": "Deep Learning",
            "seed": seed,
            "train_time_sec": res["train_time_sec"],
            "accuracy": round(res["accuracy"], 4),
            "macro_precision": round(res["macro_precision"], 4),
            "macro_recall": round(res["macro_recall"], 4),
            "macro_f1": round(res["macro_f1"], 4),
            "weighted_f1": round(res["weighted_f1"], 4)
        })
        logging.info(f"  [CNN-BiLSTM] Seed {seed}: Macro-F1 = {res['macro_f1']:.4f}")

    # 3. Multinomial Naive Bayes across 5 seeds
    logging.info("Starting Repeatability Study for Multinomial Naive Bayes...")
    for seed in SEEDS:
        seed_everything(seed)
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        X_train = vec.fit_transform(train_df["clean_text"])
        X_test = vec.transform(test_df["clean_text"])

        t0 = time.time()
        clf = MultinomialNB(alpha=0.1)
        clf.fit(X_train, train_df["label_id"])
        train_time = round(time.time() - t0, 3)

        preds = clf.predict(X_test)
        metrics = compute_metrics(test_df["label_id"].values, preds)

        raw_runs.append({
            "model": "Multinomial_Naive_Bayes",
            "family": "Traditional ML",
            "seed": seed,
            "train_time_sec": train_time,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4)
        })
        logging.info(f"  [Naive Bayes] Seed {seed}: Macro-F1 = {metrics['macro_f1']:.4f}")

    # Save raw runs
    raw_df = pd.DataFrame(raw_runs)
    raw_path = os.path.join(ROOT_DIR, "results", "repeatability_runs.csv")
    raw_df.to_csv(raw_path, index=False)
    logging.info(f"Saved raw repeatability runs to {raw_path}")

    # Compute summary statistics
    summary_rows = []
    for model_name, grp in raw_df.groupby("model"):
        f1_vals = grp["macro_f1"].values
        n = len(f1_vals)
        mean_f1 = float(np.mean(f1_vals))
        std_f1 = float(np.std(f1_vals, ddof=1)) if n > 1 else 0.0
        rsd_pct = float(std_f1 / mean_f1 * 100.0) if mean_f1 > 0 else 0.0
        min_f1 = float(np.min(f1_vals))
        max_f1 = float(np.max(f1_vals))

        if std_f1 > 0 and n > 1:
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_half = t_crit * (std_f1 / np.sqrt(n))
            ci_lower = max(0.0, mean_f1 - ci_half)
            ci_upper = min(1.0, mean_f1 + ci_half)
        else:
            ci_lower = mean_f1
            ci_upper = mean_f1

        summary_rows.append({
            "model": model_name,
            "family": grp["family"].iloc[0],
            "n_runs": n,
            "seeds": str(SEEDS),
            "mean_macro_f1": round(mean_f1, 4),
            "std_macro_f1": round(std_f1, 4),
            "rsd_percent": round(rsd_pct, 4),
            "min_macro_f1": round(min_f1, 4),
            "max_macro_f1": round(max_f1, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "ci_95_formatted": f"[{round(ci_lower, 4):.4f}, {round(ci_upper, 4):.4f}]"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(ROOT_DIR, "results", "repeatability_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved repeatability summary to {summary_path}")
    print(summary_df.to_string(index=False))
    return raw_df, summary_df


if __name__ == "__main__":
    run_repeatability_study()
