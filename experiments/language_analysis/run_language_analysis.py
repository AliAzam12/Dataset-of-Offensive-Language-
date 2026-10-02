"""
Phase 9: Language-Conditioned Performance Analysis.
Calculates performance separately across:
- Urdu
- Roman Urdu
- Pashto
- English
Computes:
- Macro-F1 per language
- Accuracy per language
- Worst-language Macro-F1
- Best-language Macro-F1
- Language performance range (max - min)
Saves:
- results/language_performance.csv
Generates:
- plots/language_performance.png and .pdf
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.evaluation.metrics import compute_metrics
from src.models.traditional_baselines import RobustRandomForest
import __main__
__main__.RobustRandomForest = RobustRandomForest
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LANGUAGES = ["Roman Urdu", "English", "Urdu", "Pashto"]


def run_language_analysis():
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    exp_dl_dir = os.path.join(ROOT_DIR, "experiments", "deep_learning")
    exp_bl_dir = os.path.join(ROOT_DIR, "experiments", "baselines")
    results_dir = os.path.join(ROOT_DIR, "results")
    plots_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Gather predictions from completed models
    models_to_evaluate = [
        {"family": "Traditional ML", "model": "Multinomial_Naive_Bayes", "ckpt": "Multinomial_Naive_Bayes.joblib"},
        {"family": "Traditional ML", "model": "Logistic_Regression", "ckpt": "Logistic_Regression.joblib"},
        {"family": "Traditional ML", "model": "Linear_SVM", "ckpt": "Linear_SVM.joblib"},
        {"family": "Traditional ML", "model": "Random_Forest", "ckpt": "Random_Forest.joblib"},
        {"family": "Deep Learning", "model": "CNN", "pred_file": "dl_cnn_test_predictions.csv"},
        {"family": "Deep Learning", "model": "BiLSTM", "pred_file": "dl_bilstm_test_predictions.csv"},
        {"family": "Deep Learning", "model": "CNN-BiLSTM", "pred_file": "dl_cnn-bilstm_test_predictions.csv"},
        {"family": "Deep Learning", "model": "Attention-BiLSTM", "pred_file": "dl_attention-bilstm_test_predictions.csv"},
    ]

    import joblib
    rows = []

    for m in models_to_evaluate:
        model_name = m["model"]
        preds = None

        if "ckpt" in m:
            ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "baselines", m["ckpt"])
            if os.path.exists(ckpt_path):
                pipeline = joblib.load(ckpt_path)
                preds = pipeline.predict(test_df["clean_text"])
        elif "pred_file" in m:
            pred_path = os.path.join(exp_dl_dir, m["pred_file"])
            if os.path.exists(pred_path):
                p_df = pd.read_csv(pred_path)
                preds = p_df["predicted_label_id"].values

        if preds is None:
            continue

        y_true = test_df["label_id"].values
        langs = test_df["language"].values

        row = {
            "family": m["family"],
            "model": model_name,
            "overall_accuracy": round(float(accuracy_score(y_true, preds)), 4),
            "overall_macro_f1": round(float(f1_score(y_true, preds, average="macro")), 4)
        }

        lang_f1s = []
        for lang in LANGUAGES:
            mask = (langs == lang)
            if np.sum(mask) > 0:
                l_acc = float(accuracy_score(y_true[mask], preds[mask]))
                l_f1 = float(f1_score(y_true[mask], preds[mask], average="macro"))
                row[f"{lang}_accuracy"] = round(l_acc, 4)
                row[f"{lang}_macro_f1"] = round(l_f1, 4)
                row[f"{lang}_samples"] = int(np.sum(mask))
                lang_f1s.append(l_f1)
            else:
                row[f"{lang}_accuracy"] = None
                row[f"{lang}_macro_f1"] = None
                row[f"{lang}_samples"] = 0

        row["worst_language_f1"] = round(min(lang_f1s), 4)
        row["best_language_f1"] = round(max(lang_f1s), 4)
        row["language_range"] = round(max(lang_f1s) - min(lang_f1s), 4)

    # Add mBERT if available in results/preds_mBERT.csv or transformer_results.csv
    preds_mbert_path = os.path.join(results_dir, "preds_mBERT.csv")
    if os.path.exists(preds_mbert_path):
        m_df = pd.read_csv(preds_mbert_path)
        y_t = m_df["label_id"].values
        y_p = m_df["predicted_label_id"].values
        l_arr = m_df["language"].values
        row_m = {
            "family": "Transformer",
            "model": "mBERT",
            "overall_accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "overall_macro_f1": round(float(f1_score(y_t, y_p, average="macro")), 4)
        }
        l_f1s = []
        for lang in LANGUAGES:
            msk = (l_arr == lang)
            if np.sum(msk) > 0:
                l_acc = float(accuracy_score(y_t[msk], y_p[msk]))
                l_f1 = float(f1_score(y_t[msk], y_p[msk], average="macro"))
                row_m[f"{lang}_accuracy"] = round(l_acc, 4)
                row_m[f"{lang}_macro_f1"] = round(l_f1, 4)
                row_m[f"{lang}_samples"] = int(np.sum(msk))
                l_f1s.append(l_f1)
            else:
                row_m[f"{lang}_accuracy"] = None
                row_m[f"{lang}_macro_f1"] = None
                row_m[f"{lang}_samples"] = 0
        row_m["worst_language_f1"] = round(min(l_f1s), 4)
        row_m["best_language_f1"] = round(max(l_f1s), 4)
        row_m["language_range"] = round(max(l_f1s) - min(l_f1s), 4)
        rows.append(row_m)

    df_lang = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "language_performance.csv")
    df_lang.to_csv(out_csv, index=False)
    logging.info(f"Saved language performance metrics to {out_csv}")
    print(df_lang[["model", "overall_macro_f1", "Roman Urdu_macro_f1", "English_macro_f1", "Urdu_macro_f1", "Pashto_macro_f1", "language_range"]].to_string(index=False))

    # Generate Publication-Grade Plot
    from experiments.generate_plots import plot_language_performance
    plot_language_performance(results_dir, plots_dir)

    return df_lang


if __name__ == "__main__":
    run_language_analysis()
