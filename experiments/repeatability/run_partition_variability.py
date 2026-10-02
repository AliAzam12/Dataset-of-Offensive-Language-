"""
Phase 11: Data-Split / Partition Variability Experiment.
Constructs 5 valid GROUP-AWARE dataset splits using split seeds: 11, 22, 33, 44, 55.
Maintains:
- Strict zero-leakage across independence_group_id
- 70% train, 10% validation, 20% test ratios
Trains representative models (Linear SVM and CNN-BiLSTM) on each partition split.
Compares:
- Model seed-to-seed variability (fixed split, varying training seed)
  vs
- Dataset partition variability (fixed training seed, varying dataset partition)
Saves:
- results/partition_variability.csv
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.evaluation.metrics import compute_metrics
from src.models.deep_learning_models import Vocab, TextDataset, TextCNN_BiLSTM, train_single_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SPLIT_SEEDS = [11, 22, 33, 44, 55]


def create_group_aware_split(full_df: pd.DataFrame, split_seed: int, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Partitions full_df into train, val, test by unique independence_group_id
    to guarantee zero cross-partition leakage.
    """
    rng = np.random.RandomState(split_seed)
    unique_groups = full_df["independence_group_id"].unique()
    rng.shuffle(unique_groups)

    n_total = len(unique_groups)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    train_df = full_df[full_df["independence_group_id"].isin(train_groups)].copy().reset_index(drop=True)
    val_df = full_df[full_df["independence_group_id"].isin(val_groups)].copy().reset_index(drop=True)
    test_df = full_df[full_df["independence_group_id"].isin(test_groups)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def run_partition_variability_study():
    full_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "full_data.csv"))
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    records = []

    for split_seed in SPLIT_SEEDS:
        logging.info(f"Generating Group-Aware Split with Seed {split_seed}...")
        train_df, val_df, test_df = create_group_aware_split(full_df, split_seed)

        # 1. Linear SVM
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        X_train = vec.fit_transform(train_df["clean_text"])
        X_test = vec.transform(test_df["clean_text"])

        t0 = time.time()
        svm = LinearSVC(C=0.1, max_iter=2000, random_state=42)
        svm.fit(X_train, train_df["label_id"])
        svm_time = round(time.time() - t0, 3)

        svm_preds = svm.predict(X_test)
        svm_metrics = compute_metrics(test_df["label_id"].values, svm_preds)

        records.append({
            "model": "Linear_SVM",
            "family": "Traditional ML",
            "split_seed": split_seed,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_time_sec": svm_time,
            "accuracy": round(svm_metrics["accuracy"], 4),
            "macro_f1": round(svm_metrics["macro_f1"], 4),
            "weighted_f1": round(svm_metrics["weighted_f1"], 4)
        })
        logging.info(f"  [SVM] Split Seed {split_seed}: Macro-F1 = {svm_metrics['macro_f1']:.4f}")

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
            TextCNN_BiLSTM, f"CNN-BiLSTM_split_{split_seed}",
            train_loader, val_loader, test_loader,
            val_y, val_langs, test_y, test_langs,
            vocab_size=vocab_size, max_epochs=8, patience=2
        )

        records.append({
            "model": "CNN-BiLSTM",
            "family": "Deep Learning",
            "split_seed": split_seed,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_time_sec": dl_res["train_time_sec"],
            "accuracy": round(dl_res["accuracy"], 4),
            "macro_f1": round(dl_res["macro_f1"], 4),
            "weighted_f1": round(dl_res["weighted_f1"], 4)
        })
        logging.info(f"  [CNN-BiLSTM] Split Seed {split_seed}: Macro-F1 = {dl_res['macro_f1']:.4f}")

    df_part = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "partition_variability.csv")
    df_part.to_csv(out_csv, index=False)
    logging.info(f"Saved partition variability results to {out_csv}")
    print(df_part.to_string(index=False))

    # Summary comparing seed vs partition variability
    print("\n--- VARIABILITY COMPARISON SUMMARY ---")
    rep_summary_path = os.path.join(results_dir, "repeatability_summary.csv")
    if os.path.exists(rep_summary_path):
        rep_df = pd.read_csv(rep_summary_path)
        for m in ["Linear_SVM", "CNN-BiLSTM"]:
            m_part = df_part[df_part["model"] == m]["macro_f1"]
            m_rep = rep_df[rep_df["model"] == m]
            rep_std = m_rep["std_macro_f1"].iloc[0] if len(m_rep) > 0 else 0.0
            print(f"Model: {m}")
            print(f"  Seed-to-Seed Variability Std:      {rep_std:.4f}")
            print(f"  Partition-to-Partition Std:        {np.std(m_part, ddof=1):.4f} (Mean: {np.mean(m_part):.4f})")

    return df_part


if __name__ == "__main__":
    run_partition_variability_study()
