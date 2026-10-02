"""
Phase 14: Model Comparison Significance Testing.
Implements paired statistical testing on identical test examples across major models:
1. McNemar Test with continuity correction and exact binomial calculation when discordance is small.
2. Paired Non-Parametric Bootstrap (B = 10,000 iterations) to compute:
   - Observed Macro-F1 Difference (Delta = F1_ModelA - F1_ModelB)
   - 95% Confidence Interval for Macro-F1 Difference [Delta_lower, Delta_upper]
   - Two-sided empirical p-value
Saves:
- results/statistical_comparisons.csv
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.models.traditional_baselines import RobustRandomForest
import __main__
__main__.RobustRandomForest = RobustRandomForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_PAIRED_BOOTSTRAP = 10000


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    Computes McNemar's test for paired classification predictions.
    Contingency table:
      b: Model A correct, Model B incorrect
      c: Model A incorrect, Model B correct
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0, b, c

    if total_discordant < 25:
        # Exact two-sided binomial test
        p_val = stats.binomtest(b, total_discordant, 0.5, alternative='two-sided').pvalue
        stat = float(b)
    else:
        # Chi-square with Edwards continuity correction
        stat = float((abs(b - c) - 1.0) ** 2 / total_discordant)
        p_val = float(stats.chi2.sf(stat, df=1))

    return round(stat, 4), round(p_val, 4), b, c


def paired_bootstrap_f1_test(y_true, y_pred_a, y_pred_b, n_iter=10000, seed=42):
    """
    Computes paired bootstrap test for Macro-F1 difference between Model A and Model B.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    obs_diff = f1_score(y_true, y_pred_a, average="macro") - f1_score(y_true, y_pred_b, average="macro")
    diffs = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        f1_a = f1_score(yt, y_pred_a[idx], average="macro", zero_division=0)
        f1_b = f1_score(yt, y_pred_b[idx], average="macro", zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided empirical p-value under null hypothesis (H0: diff = 0)
    p_val = float(np.mean(np.abs(diffs - np.mean(diffs)) >= np.abs(obs_diff)))
    return round(obs_diff, 4), round(ci_lower, 4), round(ci_upper, 4), round(p_val, 4)


def run_statistical_significance_comparisons():
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    exp_dl_dir = os.path.join(ROOT_DIR, "experiments", "deep_learning")
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    y_true = test_df["label_id"].values

    # Collect predictions for key representative models
    pred_dict = {}

    # Classical models
    for m in ["Multinomial_Naive_Bayes", "Logistic_Regression", "Linear_SVM", "Random_Forest"]:
        ckpt = os.path.join(ROOT_DIR, "checkpoints", "baselines", f"{m}.joblib")
        if os.path.exists(ckpt):
            pipe = joblib.load(ckpt)
            pred_dict[m] = pipe.predict(test_df["clean_text"])

    # Deep learning models
    for m in ["CNN", "BiLSTM", "CNN-BiLSTM", "Attention-BiLSTM"]:
        pred_csv = os.path.join(exp_dl_dir, f"dl_{m.lower()}_test_predictions.csv")
        if os.path.exists(pred_csv):
            pdf = pd.read_csv(pred_csv)
            pred_dict[m] = pdf["predicted_label_id"].values

    # Key pairwise comparisons to perform
    comparisons = [
        ("CNN-BiLSTM", "Linear_SVM", "Best Deep Learning vs Best Classical"),
        ("CNN-BiLSTM", "Multinomial_Naive_Bayes", "Best Deep Learning vs Naive Bayes"),
        ("Linear_SVM", "Logistic_Regression", "Linear SVM vs Logistic Regression"),
        ("CNN-BiLSTM", "Attention-BiLSTM", "CNN-BiLSTM vs Attention-BiLSTM"),
        ("CNN-BiLSTM", "CNN", "CNN-BiLSTM vs TextCNN"),
        ("CNN-BiLSTM", "BiLSTM", "CNN-BiLSTM vs TextBiLSTM"),
        ("Random_Forest", "Linear_SVM", "Random Forest vs Linear SVM"),
        ("Random_Forest", "CNN-BiLSTM", "Random Forest vs CNN-BiLSTM")
    ]

    records = []
    for model_a, model_b, desc in comparisons:
        if model_a not in pred_dict or model_b not in pred_dict:
            continue

        preds_a = pred_dict[model_a]
        preds_b = pred_dict[model_b]

        mcnemar_stat, mcnemar_p, b, c = mcnemar_test(y_true, preds_a, preds_b)
        obs_diff, ci_l, ci_u, boot_p = paired_bootstrap_f1_test(y_true, preds_a, preds_b, n_iter=N_PAIRED_BOOTSTRAP)

        is_significant = (mcnemar_p < 0.05) or (ci_l > 0 or ci_u < 0)

        records.append({
            "comparison": f"{model_a} vs {model_b}",
            "description": desc,
            "model_a": model_a,
            "model_b": model_b,
            "f1_a": round(f1_score(y_true, preds_a, average="macro"), 4),
            "f1_b": round(f1_score(y_true, preds_b, average="macro"), 4),
            "delta_macro_f1 (A - B)": obs_diff,
            "delta_95_ci_lower": ci_l,
            "delta_95_ci_upper": ci_u,
            "delta_95_ci": f"[{ci_l:+.4f}, {ci_u:+.4f}]",
            "paired_bootstrap_p": boot_p,
            "mcnemar_stat": mcnemar_stat,
            "mcnemar_p_value": mcnemar_p,
            "discordant_b (A correct, B wrong)": b,
            "discordant_c (A wrong, B correct)": c,
            "statistically_significant (alpha=0.05)": is_significant
        })
        logging.info(f"[{model_a} vs {model_b}]: Delta F1 = {obs_diff:+.4f} (95% CI: [{ci_l:+.4f}, {ci_u:+.4f}]) | McNemar p={mcnemar_p:.4f} (b={b}, c={c})")

    df_sig = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "statistical_comparisons.csv")
    df_sig.to_csv(out_csv, index=False)
    logging.info(f"Saved statistical comparisons to {out_csv}")
    print(df_sig[["comparison", "delta_macro_f1 (A - B)", "delta_95_ci", "mcnemar_p_value", "statistically_significant (alpha=0.05)"]].to_string(index=False))
    return df_sig


if __name__ == "__main__":
    run_statistical_significance_comparisons()
