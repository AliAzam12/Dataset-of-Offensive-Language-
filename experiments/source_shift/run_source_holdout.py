"""
Phase 10: Source Robustness / Leave-One-Source-Out (LOSO) Cross-Validation.
Evaluates domain generalization and out-of-distribution transfer by:
1. Completely excluding a target source dataset from training and validation.
2. Training on all remaining valid sources.
3. Evaluating on the held-out source.
Computes:
- In-domain (Mixed) Macro-F1
- Held-out source Macro-F1
- Absolute performance drop (In-Domain - Held-Out)
- Percentage retention (Held-Out / In-Domain * 100)
Saves:
- results/source_holdout_results.csv
Generates:
- plots/source_holdout_degradation.png and .pdf
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.evaluation.metrics import compute_metrics
from src.models.deep_learning_models import Vocab, TextDataset, TextCNN_BiLSTM, train_single_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Major sources with sufficient sample support across all 4 languages
TARGET_SOURCES = [
    "RU_CodeMixed_Private",
    "RU_Private_FB_Comments",
    "EN_Offensive_Benchmark_Private",
    "UR_ArabicScript_Private",
    "PS_LowResource_Private",
    "EN_Public_Comments",
    "PS_Private_Social_Comments",
    "UR_Private_Tweets"
]

# Baseline in-domain test Macro-F1s (from standard split)
IN_DOMAIN_F1 = {
    "Linear_SVM": 0.9917,
    "CNN-BiLSTM": 0.9959
}


def run_source_holdout():
    full_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "full_data.csv"))
    results_dir = os.path.join(ROOT_DIR, "results")
    plots_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results = []

    for src in TARGET_SOURCES:
        logging.info(f"\n==================== EVALUATING HELD-OUT SOURCE: {src} ====================")
        held_out_mask = (full_df["source_dataset"] == src)
        test_df = full_df[held_out_mask].copy().reset_index(drop=True)
        train_pool_df = full_df[~held_out_mask].copy().reset_index(drop=True)

        n_held_out = len(test_df)
        n_train_pool = len(train_pool_df)
        lang = test_df["language"].iloc[0]

        # Ensure label diversity in held-out source
        n_pos = (test_df["label_id"] == 1).sum()
        n_neg = (test_df["label_id"] == 0).sum()
        logging.info(f"Source: {src} (Lang: {lang}) | Held-Out N={n_held_out} (Offensive: {n_pos}, Non-Offensive: {n_neg}) | Train Pool N={n_train_pool}")

        if n_pos == 0 or n_neg == 0:
            logging.warning(f"Skipping {src}: only one class present in held-out set.")
            continue

        # Split train_pool into 85% train, 15% validation
        seed_everything(42)
        indices = np.random.permutation(len(train_pool_df))
        n_val = max(50, int(len(train_pool_df) * 0.15))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_df = train_pool_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = train_pool_df.iloc[val_idx].copy().reset_index(drop=True)

        # 1. Linear SVM
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        X_train = vec.fit_transform(train_df["clean_text"])
        X_test = vec.transform(test_df["clean_text"])

        svm_model = LinearSVC(C=0.1, max_iter=2000, random_state=42)
        svm_model.fit(X_train, train_df["label_id"])
        svm_preds = svm_model.predict(X_test)
        svm_f1 = float(f1_score(test_df["label_id"], svm_preds, average="macro"))
        svm_acc = float(accuracy_score(test_df["label_id"], svm_preds))

        in_f1_svm = IN_DOMAIN_F1["Linear_SVM"]
        drop_svm = round(in_f1_svm - svm_f1, 4)
        retention_svm = round((svm_f1 / in_f1_svm) * 100.0, 2)

        results.append({
            "held_out_source": src,
            "language": lang,
            "model": "Linear_SVM",
            "family": "Traditional ML",
            "held_out_samples": n_held_out,
            "train_samples": len(train_df),
            "in_domain_macro_f1": in_f1_svm,
            "held_out_macro_f1": round(svm_f1, 4),
            "held_out_accuracy": round(svm_acc, 4),
            "absolute_f1_drop": drop_svm,
            "retention_percent": retention_svm
        })
        logging.info(f"  [Linear SVM] In-Domain: {in_f1_svm:.4f} -> Held-Out: {svm_f1:.4f} (Drop: {drop_svm:+.4f}, Ret: {retention_svm:.1f}%)")

        # 2. CNN-BiLSTM
        vocab = Vocab(max_size=5000)
        vocab.build_vocab(train_df["clean_text"])
        vocab_size = len(vocab.w2i)

        train_ds = TextDataset(train_df["clean_text"].tolist(), train_df["label_id"].tolist(), vocab, max_len=64)
        val_ds = TextDataset(val_df["clean_text"].tolist(), val_df["label_id"].tolist(), vocab, max_len=64)
        test_ds = TextDataset(test_df["clean_text"].tolist(), test_df["label_id"].tolist(), vocab, max_len=64)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        val_y = val_df["label_id"].values
        val_langs = val_df["language"].values
        test_y = test_df["label_id"].values
        test_langs = test_df["language"].values

        dl_res = train_single_model(
            TextCNN_BiLSTM, f"CNN-BiLSTM_LOSO_{src}",
            train_loader, val_loader, test_loader,
            val_y, val_langs, test_y, test_langs,
            vocab_size=vocab_size, max_epochs=8, patience=2
        )
        dl_f1 = dl_res["macro_f1"]
        dl_acc = dl_res["accuracy"]

        in_f1_dl = IN_DOMAIN_F1["CNN-BiLSTM"]
        drop_dl = round(in_f1_dl - dl_f1, 4)
        retention_dl = round((dl_f1 / in_f1_dl) * 100.0, 2)

        results.append({
            "held_out_source": src,
            "language": lang,
            "model": "CNN-BiLSTM",
            "family": "Deep Learning",
            "held_out_samples": n_held_out,
            "train_samples": len(train_df),
            "in_domain_macro_f1": in_f1_dl,
            "held_out_macro_f1": round(dl_f1, 4),
            "held_out_accuracy": round(dl_acc, 4),
            "absolute_f1_drop": drop_dl,
            "retention_percent": retention_dl
        })
        logging.info(f"  [CNN-BiLSTM] In-Domain: {in_f1_dl:.4f} -> Held-Out: {dl_f1:.4f} (Drop: {drop_dl:+.4f}, Ret: {retention_dl:.1f}%)")

    # Save results
    df_loso = pd.DataFrame(results)
    out_csv = os.path.join(results_dir, "source_holdout_results.csv")
    df_loso.to_csv(out_csv, index=False)
    logging.info(f"Saved source holdout results to {out_csv}")
    print(df_loso[["held_out_source", "language", "model", "in_domain_macro_f1", "held_out_macro_f1", "absolute_f1_drop", "retention_percent"]].to_string(index=False))

    # Generate publication-grade degradation plot
    from experiments.generate_plots import plot_source_holdout_degradation
    plot_source_holdout_degradation(results_dir, plots_dir)

    return df_loso


if __name__ == "__main__":
    run_source_holdout()
