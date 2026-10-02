"""
Phase 12: Ablation Experiments.
Evaluates the causal effect of individual design decisions against a full controlled baseline.
Changes ONE variable at a time:
  A. Full configuration (Controlled Baseline)
  B. Without class weighting (class_weight=None)
  C. Without sublinear term frequency scaling (sublinear_tf=False)
  D. Without text cleaning / preprocessing: using raw 'text' instead of 'clean_text'
  E. Word n-grams only (no bi-grams: ngram_range=(1, 1))
Computes:
- Test Macro-F1
- Delta Macro-F1 = Ablated_F1 - Full_Config_F1
Saves:
- results/ablation_results.csv
Generates:
- plots/ablation_results.png and .pdf
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_ablations():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "validation.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    results_dir = os.path.join(ROOT_DIR, "results")
    plots_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    ablations = [
        {
            "name": "Full_Configuration",
            "description": "Baseline: clean_text + class_weight='balanced' + sublinear_tf=True + ngram(1,2)",
            "text_col": "clean_text",
            "class_weight": "balanced",
            "sublinear_tf": True,
            "ngram_range": (1, 2)
        },
        {
            "name": "Without_Class_Weighting",
            "description": "Ablate class weighting (class_weight=None)",
            "text_col": "clean_text",
            "class_weight": None,
            "sublinear_tf": True,
            "ngram_range": (1, 2)
        },
        {
            "name": "Without_Sublinear_TF",
            "description": "Ablate sublinear TF scaling (sublinear_tf=False)",
            "text_col": "clean_text",
            "class_weight": "balanced",
            "sublinear_tf": False,
            "ngram_range": (1, 2)
        },
        {
            "name": "Raw_Text_Instead_Of_Clean_Text",
            "description": "Ablate text normalization: evaluate on raw 'text' instead of 'clean_text'",
            "text_col": "text",
            "class_weight": "balanced",
            "sublinear_tf": True,
            "ngram_range": (1, 2)
        },
        {
            "name": "Unigrams_Only_No_Bigrams",
            "description": "Ablate bigrams: ngram_range=(1, 1)",
            "text_col": "clean_text",
            "class_weight": "balanced",
            "sublinear_tf": True,
            "ngram_range": (1, 1)
        }
    ]

    seed_everything(42)
    rows = []
    full_macro_f1 = None

    for abl in ablations:
        text_col = abl["text_col"]
        vec = TfidfVectorizer(
            ngram_range=abl["ngram_range"],
            max_features=5000,
            sublinear_tf=abl["sublinear_tf"]
        )
        X_train = vec.fit_transform(train_df[text_col])
        X_test = vec.transform(test_df[text_col])

        t0 = time.time()
        clf = LinearSVC(C=0.1, max_iter=2000, class_weight=abl["class_weight"], random_state=42)
        clf.fit(X_train, train_df["label_id"])
        train_time = round(time.time() - t0, 3)

        preds = clf.predict(X_test)
        metrics = compute_metrics(test_df["label_id"].values, preds)

        macro_f1 = metrics["macro_f1"]
        if abl["name"] == "Full_Configuration":
            full_macro_f1 = macro_f1
            delta_f1 = 0.0
        else:
            delta_f1 = round(macro_f1 - full_macro_f1, 4)

        rows.append({
            "ablation_condition": abl["name"],
            "description": abl["description"],
            "accuracy": round(metrics["accuracy"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "macro_f1": round(macro_f1, 4),
            "delta_macro_f1": delta_f1,
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "train_time_sec": train_time
        })
        logging.info(f"Ablation [{abl['name']}]: Macro-F1 = {macro_f1:.4f} (Delta = {delta_f1:+.4f})")

    df_abl = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "ablation_results.csv")
    df_abl.to_csv(out_csv, index=False)
    logging.info(f"Saved ablation results to {out_csv}")
    print(df_abl[["ablation_condition", "macro_f1", "delta_macro_f1", "accuracy"]].to_string(index=False))

    # Generate publication-grade ablation plot
    from experiments.generate_plots import plot_ablation_results
    plot_ablation_results(results_dir, plots_dir)

    return df_abl


if __name__ == "__main__":
    run_ablations()
