"""
Phase 13: Bootstrap Uncertainty Estimation.
Performs non-parametric bootstrap resampling (B = 1000 iterations) on the fixed test set
predictions and labels for all evaluated models.
Computes:
- Point estimate of Macro-F1
- Point estimate of Accuracy
- 95% Bootstrap Confidence Interval [2.5th percentile, 97.5th percentile] for Macro-F1
- 95% Bootstrap Confidence Interval for Accuracy
- Standard Error (bootstrap std)
Saves:
- results/bootstrap_confidence_intervals.csv
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.models.traditional_baselines import RobustRandomForest
import __main__
__main__.RobustRandomForest = RobustRandomForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_BOOTSTRAP = 1000


def run_bootstrap_uncertainty():
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    exp_dl_dir = os.path.join(ROOT_DIR, "experiments", "deep_learning")
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    models_to_bootstrap = [
        {"name": "Multinomial_Naive_Bayes", "family": "Traditional ML", "ckpt": "Multinomial_Naive_Bayes.joblib"},
        {"name": "Logistic_Regression", "family": "Traditional ML", "ckpt": "Logistic_Regression.joblib"},
        {"name": "Linear_SVM", "family": "Traditional ML", "ckpt": "Linear_SVM.joblib"},
        {"name": "Random_Forest", "family": "Traditional ML", "ckpt": "Random_Forest.joblib"},
        {"name": "CNN", "family": "Deep Learning", "pred_file": "dl_cnn_test_predictions.csv"},
        {"name": "BiLSTM", "family": "Deep Learning", "pred_file": "dl_bilstm_test_predictions.csv"},
        {"name": "CNN-BiLSTM", "family": "Deep Learning", "pred_file": "dl_cnn-bilstm_test_predictions.csv"},
        {"name": "Attention-BiLSTM", "family": "Deep Learning", "pred_file": "dl_attention-bilstm_test_predictions.csv"},
    ]

    y_true = test_df["label_id"].values
    n_samples = len(y_true)
    seed_everything(42)

    # Pre-generate 1000 bootstrap sample index arrays for strict comparability
    rng = np.random.RandomState(42)
    boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(N_BOOTSTRAP)]

    records = []

    for m in models_to_bootstrap:
        name = m["name"]
        preds = None

        if "ckpt" in m:
            p = os.path.join(ROOT_DIR, "checkpoints", "baselines", m["ckpt"])
            if os.path.exists(p):
                pipe = joblib.load(p)
                preds = pipe.predict(test_df["clean_text"])
        elif "pred_file" in m:
            p = os.path.join(exp_dl_dir, m["pred_file"])
            if os.path.exists(p):
                pdf = pd.read_csv(p)
                preds = pdf["predicted_label_id"].values

        if preds is None:
            continue

        point_f1 = float(f1_score(y_true, preds, average="macro"))
        point_acc = float(accuracy_score(y_true, preds))

        boot_f1s = []
        boot_accs = []
        for idx in boot_indices:
            yt_b = y_true[idx]
            yp_b = preds[idx]
            boot_f1s.append(f1_score(yt_b, yp_b, average="macro"))
            boot_accs.append(accuracy_score(yt_b, yp_b))

        boot_f1s = np.array(boot_f1s)
        boot_accs = np.array(boot_accs)

        f1_lower = float(np.percentile(boot_f1s, 2.5))
        f1_upper = float(np.percentile(boot_f1s, 97.5))
        f1_se = float(np.std(boot_f1s, ddof=1))

        acc_lower = float(np.percentile(boot_accs, 2.5))
        acc_upper = float(np.percentile(boot_accs, 97.5))
        acc_se = float(np.std(boot_accs, ddof=1))

        records.append({
            "model": name,
            "family": m["family"],
            "bootstrap_iterations": N_BOOTSTRAP,
            "point_macro_f1": round(point_f1, 4),
            "macro_f1_se": round(f1_se, 4),
            "macro_f1_95_ci_lower": round(f1_lower, 4),
            "macro_f1_95_ci_upper": round(f1_upper, 4),
            "macro_f1_95_ci": f"[{round(f1_lower, 4):.4f}, {round(f1_upper, 4):.4f}]",
            "point_accuracy": round(point_acc, 4),
            "accuracy_se": round(acc_se, 4),
            "accuracy_95_ci_lower": round(acc_lower, 4),
            "accuracy_95_ci_upper": round(acc_upper, 4),
            "accuracy_95_ci": f"[{round(acc_lower, 4):.4f}, {round(acc_upper, 4):.4f}]"
        })
        logging.info(f"{name:25s} | Macro-F1: {point_f1:.4f} (95% CI: [{f1_lower:.4f}, {f1_upper:.4f}]) | Acc: {point_acc:.4f}")

    df_ci = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "bootstrap_confidence_intervals.csv")
    df_ci.to_csv(out_csv, index=False)
    logging.info(f"Saved bootstrap confidence intervals to {out_csv}")
    print(df_ci[["model", "point_macro_f1", "macro_f1_95_ci", "point_accuracy", "accuracy_95_ci"]].to_string(index=False))
    return df_ci


if __name__ == "__main__":
    run_bootstrap_uncertainty()
