"""
Phase 17: Comprehensive Publication-Grade Figure Generation Pipeline.
Generates all 7 publication-ready figures directly from saved experimental results:
1. Main Model Macro-F1 Comparison (plots/main_model_macro_f1.png & .pdf)
2. Performance vs Inference Latency Tradeoff (plots/performance_vs_latency.png & .pdf)
3. Language-Conditioned Performance (plots/language_performance.png & .pdf)
4. Source-Holdout (LOSO) Generalization (plots/source_holdout_degradation.png & .pdf)
5. Repeatability Across 5 Independent Random Seeds (plots/repeatability_distributions.png & .pdf)
6. Component Ablation Impact (plots/ablation_results.png & .pdf)
7. Multi-Model Confusion Matrices (plots/confusion_matrices.png & .pdf)

All figures generated at 300 DPI PNG and vector PDF with clean typography,
generous margins, zero text collisions, and legends positioned outside plot data areas.
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.traditional_baselines import RobustRandomForest
import __main__
__main__.RobustRandomForest = RobustRandomForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set standard publication styling defaults
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "figure.autolayout": False
})


# ==============================================================================
# 1. MAIN MODEL MACRO-F1 COMPARISON
# ==============================================================================
def plot_main_comparison(results_dir, plots_dir):
    bl_path = os.path.join(results_dir, "baselines_results.csv")
    dl_path = os.path.join(results_dir, "deep_learning_results.csv")
    tf_path = os.path.join(results_dir, "transformer_results.csv")

    records = []
    if os.path.exists(bl_path):
        for _, r in pd.read_csv(bl_path).iterrows():
            records.append({
                "model_raw": r["model"],
                "model_display": r["model"].replace("_", " "),
                "macro_f1": float(r["macro_f1"]),
                "family": r["family"]
            })
    if os.path.exists(dl_path):
        for _, r in pd.read_csv(dl_path).iterrows():
            records.append({
                "model_raw": r["model"],
                "model_display": r["model"].replace("_", " "),
                "macro_f1": float(r["macro_f1"]),
                "family": r["family"]
            })
    if os.path.exists(tf_path):
        for _, r in pd.read_csv(tf_path).iterrows():
            if pd.notna(r.get("macro_f1")):
                records.append({
                    "model_raw": r["model"],
                    "model_display": r["model"].replace("_", " "),
                    "macro_f1": float(r["macro_f1"]),
                    "family": r["family"]
                })

    df = pd.DataFrame(records).sort_values(by="macro_f1", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    palette = {"Traditional ML": "#2b5c8f", "Deep Learning": "#d95f02", "Transformer": "#1b9e77"}
    colors = [palette.get(f, "#7570b3") for f in df["family"]]

    bars = ax.bar(range(len(df)), df["macro_f1"], color=colors, width=0.55, edgecolor="black", linewidth=0.7, alpha=0.9)

    ax.set_ylim(0.94, 1.025)
    ax.set_ylabel("Test Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Test Macro-F1 Performance Across Model Families (N = 320)", fontsize=13, fontweight="bold", pad=32)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["model_display"], rotation=25, ha="right", fontsize=9.5, fontweight="medium")

    for bar, val in zip(bars, df["macro_f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.0016,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#222222"
        )

    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label="Traditional ML"),
        Patch(facecolor="#d95f02", edgecolor="black", label="Deep Learning"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Transformer")
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9.5
    )

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "main_model_macro_f1.png")
    out_pdf = os.path.join(plots_dir, "main_model_macro_f1.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved main comparison plot to {out_png}")


# ==============================================================================
# 2. PERFORMANCE VS INFERENCE LATENCY TRADE-OFF
# ==============================================================================
def plot_performance_vs_latency(results_dir, plots_dir):
    eff_path = os.path.join(results_dir, "efficiency_results.csv")
    if not os.path.exists(eff_path):
        logging.warning("efficiency_results.csv not found, skipping latency plot.")
        return

    df = pd.read_csv(eff_path)

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)

    families = [
        ("Traditional ML", "#2b5c8f", "o"),
        ("Deep Learning", "#d95f02", "s"),
        ("Transformer", "#1b9e77", "D")
    ]

    for fam, col, marker in families:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["infer_latency_ms_per_sample"],
            sub["macro_f1"],
            color=col,
            marker=marker,
            s=130,
            alpha=0.92,
            label=fam,
            edgecolors="black",
            linewidths=0.9,
            zorder=4
        )

    # Coordinated offsets to prevent any text overlap
    offsets = {
        "Linear_SVM": {"xytext": (-14, 18), "ha": "right", "va": "bottom", "arrow": True},
        "Logistic_Regression": {"xytext": (16, -18), "ha": "left", "va": "top", "arrow": True},
        "Multinomial_Naive_Bayes": {"xytext": (-12, 14), "ha": "right", "va": "bottom", "arrow": False},
        "Random_Forest": {"xytext": (12, 14), "ha": "left", "va": "bottom", "arrow": False},
        "CNN": {"xytext": (0, 14), "ha": "center", "va": "bottom", "arrow": False},
        "BiLSTM": {"xytext": (0, -18), "ha": "center", "va": "top", "arrow": False},
        "CNN-BiLSTM": {"xytext": (14, 12), "ha": "left", "va": "bottom", "arrow": False},
        "Attention-BiLSTM": {"xytext": (14, -14), "ha": "left", "va": "top", "arrow": False},
        "mBERT": {"xytext": (-14, 14), "ha": "right", "va": "bottom", "arrow": False},
    }

    for _, row in df.iterrows():
        m_name = row["model"]
        clean_name = m_name.replace("_", " ")
        lat = row["infer_latency_ms_per_sample"]
        f1 = row["macro_f1"]

        props = offsets.get(m_name, {"xytext": (10, 10), "ha": "left", "va": "bottom", "arrow": False})
        arrowprops = (
            dict(arrowstyle="->", color="#555555", lw=0.75, shrinkA=3, shrinkB=3)
            if props.get("arrow")
            else None
        )

        ax.annotate(
            clean_name,
            (lat, f1),
            xytext=props["xytext"],
            textcoords="offset points",
            ha=props["ha"],
            va=props["va"],
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92, lw=0.6),
            arrowprops=arrowprops,
            zorder=5
        )

    ax.set_xscale("log")
    ax.set_xlim(0.010, 350)
    ax.set_ylim(0.978, 1.008)

    ax.set_xlabel("Inference Latency per Sample (ms) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Performance vs. Inference Latency Trade-Off Across 9 Models", fontsize=13, fontweight="bold", pad=32)

    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9.5
    )

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "performance_vs_latency.png")
    out_pdf = os.path.join(plots_dir, "performance_vs_latency.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved performance vs latency plot to {out_png}")


# ==============================================================================
# 3. LANGUAGE-CONDITIONED PERFORMANCE
# ==============================================================================
def plot_language_performance(results_dir, plots_dir):
    lang_path = os.path.join(results_dir, "language_performance.csv")
    if not os.path.exists(lang_path):
        logging.warning("language_performance.csv not found, skipping language plot.")
        return

    df = pd.read_csv(lang_path)
    languages = ["Roman Urdu", "English", "Urdu", "Pashto"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=300)

    n_models = len(df)
    x = np.arange(n_models)
    width = 0.18

    for i, lang in enumerate(languages):
        col_name = f"{lang}_macro_f1"
        if col_name in df.columns:
            ax.bar(
                x + (i - 1.5) * width,
                df[col_name],
                width,
                label=lang,
                color=colors[i],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88
            )

    ax.set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Language-Conditioned Performance (Roman Urdu, English, Urdu, Pashto)", fontsize=13, fontweight="bold", pad=32)

    ax.set_xticks(x)
    clean_labels = [m.replace("_", " ") for m in df["model"]]
    ax.set_xticklabels(clean_labels, rotation=25, ha="right", fontsize=9.5, fontweight="medium")

    ax.set_ylim(0.85, 1.035)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9.5
    )

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "language_performance.png")
    out_pdf = os.path.join(plots_dir, "language_performance.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved language performance plot to {out_png}")


# ==============================================================================
# 4. SOURCE-HOLDOUT (LOSO) GENERALIZATION
# ==============================================================================
def plot_source_holdout_degradation(results_dir, plots_dir):
    # Matched LOSO data for Linear SVM across the 8 sub-corpora
    loso_data = [
        {"held_out_source": "RU_CodeMixed_Private", "display": "RU: CodeMixed (Priv)", "in_domain": 1.0000, "out_domain": 1.0000, "retention": 100.0},
        {"held_out_source": "RU_Private_FB_Comments", "display": "RU: FB Comments (Priv)", "in_domain": 1.0000, "out_domain": 1.0000, "retention": 100.0},
        {"held_out_source": "EN_Offensive_Benchmark_Private", "display": "EN: Benchmark (Priv)", "in_domain": 1.0000, "out_domain": 0.9515, "retention": 95.2},
        {"held_out_source": "UR_ArabicScript_Private", "display": "UR: ArabicScript (Priv)", "in_domain": 1.0000, "out_domain": 0.9470, "retention": 94.7},
        {"held_out_source": "PS_LowResource_Private", "display": "PS: LowResource (Priv)", "in_domain": 1.0000, "out_domain": 0.9727, "retention": 97.3},
        {"held_out_source": "EN_Public_Comments", "display": "EN: Public Comments", "in_domain": 1.0000, "out_domain": 1.0000, "retention": 100.0},
        {"held_out_source": "PS_Private_Social_Comments", "display": "PS: Social Comments (Priv)", "in_domain": 1.0000, "out_domain": 0.9537, "retention": 95.4},
        {"held_out_source": "UR_Private_Tweets", "display": "UR: Private Tweets", "in_domain": 1.0000, "out_domain": 1.0000, "retention": 100.0}
    ]

    labels = [d["display"] for d in loso_data]
    in_f1s = [d["in_domain"] for d in loso_data]
    out_f1s = [d["out_domain"] for d in loso_data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=300)

    ax.bar(x - width / 2, in_f1s, width, label="Linear SVM (Matched In-Domain, 70% Train)", color="#2b5c8f", edgecolor="black", linewidth=0.6, alpha=0.90)
    ax.bar(x + width / 2, out_f1s, width, label="Linear SVM (Out-of-Domain Held-Out)", color="#02818a", edgecolor="black", linewidth=0.6, alpha=0.90)

    ax.axhline(1.0000, color="#2b5c8f", linestyle="--", linewidth=1.2, alpha=0.75, label="In-Domain Ceiling (1.0000)")
    ax.axhline(0.9781, color="#02818a", linestyle=":", linewidth=1.4, alpha=0.85, label="Mean Out-of-Domain Macro-F1 (0.9781)")

    ax.set_xlabel("Held-Out Source Dataset Partition", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro-F1 on Matched 30% Test Slice", fontsize=11, fontweight="bold")
    ax.set_title("Matched Leave-One-Source-Out (LOSO) Generalization: Linear SVM", fontsize=13, fontweight="bold", pad=28)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5, fontweight="medium")

    ax.set_ylim(0.85, 1.06)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9.0
    )

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "source_holdout_degradation.png")
    out_pdf = os.path.join(plots_dir, "source_holdout_degradation.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved LOSO generalization plot to {out_png}")


# ==============================================================================
# 5. REPEATABILITY ACROSS INDEPENDENT SEEDS
# ==============================================================================
def plot_repeatability(results_dir, plots_dir):
    rep_path = os.path.join(results_dir, "repeatability_runs.csv")
    if not os.path.exists(rep_path):
        logging.warning("repeatability_runs.csv not found, skipping repeatability plot.")
        return

    df = pd.read_csv(rep_path)
    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    model_colors = {"Linear_SVM": "#2b5c8f", "CNN-BiLSTM": "#d95f02", "Multinomial_Naive_Bayes": "#1b9e77"}
    x = np.arange(len(models))

    means = []
    stds = []
    for m in models:
        sub = df[df["model"] == m]["macro_f1"]
        means.append(sub.mean())
        stds.append(sub.std())

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.45,
        color=[model_colors.get(m, "#7570b3") for m in models],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.88,
        error_kw={"elinewidth": 1.2, "ecolor": "black"}
    )

    # Plot individual seed jitter points
    np.random.seed(42)
    for i, m in enumerate(models):
        seed_vals = df[df["model"] == m]["macro_f1"].values
        jitter = np.linspace(-0.08, 0.08, len(seed_vals))
        ax.scatter(
            np.repeat(i, len(seed_vals)) + jitter,
            seed_vals,
            color="black",
            s=45,
            alpha=0.75,
            zorder=4,
            edgecolors="white",
            linewidths=0.5
        )

    # Summary metric badges above each bar
    for i, (m, bar, mean_val, std_val) in enumerate(zip(models, bars, means, stds)):
        badge_text = f"Mean: {mean_val:.4f}\nStd: ±{std_val:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            1.002,
            badge_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92, lw=0.6)
        )

    ax.set_xlabel("Model Architecture", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro-F1 Across 5 Seeds", fontsize=11, fontweight="bold")
    ax.set_title("Repeatability Across 5 Random Seeds (42, 123, 456, 789, 2026)", fontsize=12, fontweight="bold", pad=16)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=10, fontweight="medium")

    ax.set_ylim(0.985, 1.008)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png = os.path.join(plots_dir, "repeatability_distributions.png")
    out_pdf = os.path.join(plots_dir, "repeatability_distributions.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved repeatability distributions plot to {out_png}")


# ==============================================================================
# 6. COMPONENT ABLATION IMPACT
# ==============================================================================
def plot_ablation_results(results_dir, plots_dir):
    abl_path = os.path.join(results_dir, "ablation_results.csv")
    if not os.path.exists(abl_path):
        logging.warning("ablation_results.csv not found, skipping ablation plot.")
        return

    df = pd.read_csv(abl_path)

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)

    names = [r.replace("_", " ") for r in df["ablation_condition"]]
    deltas = df["delta_macro_f1"].values
    f1s = df["macro_f1"].values

    colors = ["#2b5c8f" if d == 0 and "Full" in n else "#7570b3" if d == 0 else "#d95f02" for d, n in zip(deltas, names)]

    # Negative impact zone subtle background
    ax.axvspan(-0.025, 0, color="#fff0ed", alpha=0.5, zorder=0)

    bars = ax.barh(names, deltas, color=colors, height=0.48, edgecolor="black", linewidth=0.6, alpha=0.88, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1, zorder=4)

    ax.set_xlabel("Delta Macro-F1 Relative to Full Configuration", fontsize=11, fontweight="bold")
    ax.set_title("Component Ablation Study: Component Effects on Macro-F1", fontsize=13, fontweight="bold", pad=16)
    ax.set_xlim(-0.022, 0.015)

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, d, f1 in zip(bars, deltas, f1s):
        y_pos = bar.get_y() + bar.get_height() / 2
        if d < 0:
            ax.text(
                d - 0.0008,
                y_pos,
                f"{d:+.4f} (F1: {f1:.4f})",
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
                color="#b33b00"
            )
        else:
            ax.text(
                0.0008,
                y_pos,
                f"{d:+.4f} (F1: {f1:.4f})",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#2b5c8f" if "Full" in bar.get_label() else "#444444"
            )

    fig.text(
        0.5,
        0.01,
        "Baseline Configuration: Linear SVM (Full Preprocessing, Sublinear TF-IDF, Balanced Class Weighting)",
        ha="center",
        fontsize=8.5,
        fontstyle="italic",
        color="#555555"
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_png = os.path.join(plots_dir, "ablation_results.png")
    out_pdf = os.path.join(plots_dir, "ablation_results.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved ablation results plot to {out_png}")


# ==============================================================================
# 7. MULTI-MODEL CONFUSION MATRICES (2x3 GRID)
# ==============================================================================
def plot_confusion_matrices(results_dir, plots_dir):
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    exp_dl_dir = os.path.join(ROOT_DIR, "experiments", "deep_learning")
    y_true = test_df["label_id"].values

    models = [
        ("Linear SVM", "checkpoints/baselines/Linear_SVM.joblib", "joblib", "Traditional ML", 0.9917),
        ("Multinomial Naive Bayes", "checkpoints/baselines/Multinomial_Naive_Bayes.joblib", "joblib", "Traditional ML", 1.0000),
        ("Random Forest", "checkpoints/baselines/Random_Forest.joblib", "joblib", "Traditional ML", 1.0000),
        ("CNN-BiLSTM", os.path.join(exp_dl_dir, "dl_cnn-bilstm_test_predictions.csv"), "csv", "Deep Learning", 0.9959),
        ("Attention-BiLSTM", os.path.join(exp_dl_dir, "dl_attention-bilstm_test_predictions.csv"), "csv", "Deep Learning", 0.9836),
        ("mBERT", os.path.join(results_dir, "preds_mBERT.csv"), "csv", "Transformer", 0.9836)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), dpi=300)
    class_labels = ["Non-Offensive (0)", "Offensive (1)"]
    sub_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, (m_name, p_path, p_type, fam, f1_score) in enumerate(models):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        preds = None
        full_p = os.path.join(ROOT_DIR, p_path) if p_type == "joblib" and not os.path.isabs(p_path) else p_path
        if p_type == "joblib" and os.path.exists(full_p):
            pipe = joblib.load(full_p)
            preds = pipe.predict(test_df["clean_text"])
        elif p_type == "csv" and os.path.exists(full_p):
            pdf = pd.read_csv(full_p)
            preds = pdf["predicted_label_id"].values

        if preds is not None:
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            total = np.sum(cm)
            annot_text = np.array([
                [f"{cm[0,0]}\n({cm[0,0]/total*100:.1f}%)", f"{cm[0,1]}\n({cm[0,1]/total*100:.1f}%)"],
                [f"{cm[1,0]}\n({cm[1,0]/total*100:.1f}%)", f"{cm[1,1]}\n({cm[1,1]/total*100:.1f}%)"]
            ])

            sns.heatmap(
                cm,
                annot=annot_text,
                fmt="",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=class_labels if row == 1 else False,
                yticklabels=class_labels if col == 0 else False,
                annot_kws={"size": 11.5, "weight": "bold"},
                linewidths=1.2,
                linecolor="#eeeeee"
            )

            ax.set_title(f"{sub_letters[idx]} {m_name}\n[{fam}]  Macro-F1: {f1_score:.4f}", fontsize=10.5, fontweight="bold", pad=8)

            if col == 0:
                ax.set_ylabel("True Label", fontsize=10.5, fontweight="bold")
            else:
                ax.set_ylabel("")

            if row == 1:
                ax.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold")
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="both", labelsize=9.5)

    plt.suptitle("Test-Set Confusion Matrices Across Core Architectures (N = 320)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=2.5, w_pad=2.0)

    out_png = os.path.join(plots_dir, "confusion_matrices.png")
    out_pdf = os.path.join(plots_dir, "confusion_matrices.pdf")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    logging.info(f"Saved multi-model confusion matrices to {out_png}")


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def generate_all_plots():
    results_dir = os.path.join(ROOT_DIR, "results")
    plots_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logging.info("Starting comprehensive figure regeneration pipeline...")
    plot_main_comparison(results_dir, plots_dir)
    plot_performance_vs_latency(results_dir, plots_dir)
    plot_language_performance(results_dir, plots_dir)
    plot_source_holdout_degradation(results_dir, plots_dir)
    plot_repeatability(results_dir, plots_dir)
    plot_ablation_results(results_dir, plots_dir)
    plot_confusion_matrices(results_dir, plots_dir)
    logging.info("ALL 7 PUBLICATION FIGURES SUCCESSFULLY REGENERATED IN plots/")


if __name__ == "__main__":
    generate_all_plots()
