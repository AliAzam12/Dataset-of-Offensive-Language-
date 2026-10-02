"""
experiments/generate_architecture_diagram.py
---------------------------------------------
Generates a true Q1 journal-grade (IEEE TPAMI / Springer JIIS) architecture diagram.
Purely visual, modular block diagram with minimal wording and explicit tensor/layer flow.

Outputs:
  - plots/system_architecture.png (300 DPI)
  - plots/system_architecture.pdf (Vector Graphics)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle=None, tag=None,
             bg_color="#ffffff", border_color="#1e293b", title_color="#0f172a",
             sub_color="#475569", tag_bg="#e2e8f0", tag_fg="#1e293b",
             radius=0.016, lw=1.3, zorder=3):
    """Draws a sleek, modern Q1 scientific module block with minimal wording."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)

    # Optional tag badge (e.g. dimensions)
    if tag:
        ax.text(
            x + w / 2, y + h - 0.016, tag,
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=tag_fg,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_bg, edgecolor="none"),
            zorder=zorder + 1
        )

    # Title & Subtitle positioning
    if subtitle:
        title_y = y + h / 2 + (0.012 if not tag else 0.002)
        sub_y = y + h / 2 - (0.016 if not tag else 0.024)
        ax.text(x + w / 2, title_y, title, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=title_color, zorder=zorder + 1)
        ax.text(x + w / 2, sub_y, subtitle, ha="center", va="center",
                fontsize=7.4, fontstyle="normal", color=sub_color, zorder=zorder + 1)
    else:
        title_y = y + h / 2 - (0.008 if tag else 0)
        ax.text(x + w / 2, title_y, title, ha="center", va="center",
                fontsize=8.8, fontweight="bold", color=title_color, zorder=zorder + 1)


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.8, label=None, label_bg="#ffffff", zorder=5):
    """Draws a bold, authoritative scientific flow arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        color=color, linewidth=lw,
        mutation_scale=12, zorder=zorder
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2, label,
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.008", facecolor=label_bg, edgecolor=color, lw=0.8),
            zorder=zorder + 1
        )


def generate_q1_architecture():
    fig, ax = plt.subplots(figsize=(15.5, 6.2), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # =========================================================================
    # SECTION REGION LABELS (Subtle, sleek, non-intrusive)
    # =========================================================================
    # Phase A
    ax.text(0.125, 0.94, "(A) DATA INGESTION & SPLITTING", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#475569")
    reg_a = FancyBboxPatch((0.020, 0.05), 0.210, 0.86, boxstyle="round,pad=0.008,rounding_size=0.02",
                           facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0, linestyle="--", zorder=1)
    ax.add_patch(reg_a)

    # Phase B
    ax.text(0.505, 0.94, "(B) MULTI-PARADIGM MODELING BACKBONES", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#475569")
    reg_b = FancyBboxPatch((0.245, 0.05), 0.520, 0.86, boxstyle="round,pad=0.008,rounding_size=0.02",
                           facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=1.0, linestyle="--", zorder=1)
    ax.add_patch(reg_b)

    # Phase C
    ax.text(0.880, 0.94, "(C) CLASSIFICATION", ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="#475569")
    reg_c = FancyBboxPatch((0.780, 0.05), 0.200, 0.86, boxstyle="round,pad=0.008,rounding_size=0.02",
                           facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0, linestyle="--", zorder=1)
    ax.add_patch(reg_c)

    # =========================================================================
    # (A) INGESTION & GROUP PARTITIONING (Left Column)
    # =========================================================================
    # 1. Multilingual Corpus
    draw_box(ax, 0.035, 0.64, 0.180, 0.20,
             "Multilingual Corpus",
             "Urdu · Pashto · Roman · English",
             tag="N = 11,789",
             bg_color="#f1f5f9", border_color="#94a3b8", title_color="#0f172a",
             tag_bg="#e2e8f0", tag_fg="#334155")

    # Downward arrow to Sanitization / Grouping
    draw_arrow(ax, 0.125, 0.64, 0.125, 0.49, color="#0f766e", lw=2.0,
               label="Clean & Group")

    # 2. Canonical Group Partition
    draw_box(ax, 0.035, 0.28, 0.180, 0.21,
             "Canonical Group Partition",
             "1,601 Disjoint Groups\nZero Overlap Across 5 Reps",
             tag="Audited Disjoint",
             bg_color="#ecfdf5", border_color="#10b981", title_color="#065f46",
             sub_color="#047857", tag_bg="#d1fae5", tag_fg="#047857")

    # Sub-split badges below (Train / Val / Test)
    draw_box(ax, 0.035, 0.10, 0.056, 0.11, "Train", "70%", bg_color="#ffffff", border_color="#10b981", lw=1.0)
    draw_box(ax, 0.097, 0.10, 0.056, 0.11, "Val", "10%", bg_color="#ffffff", border_color="#10b981", lw=1.0)
    draw_box(ax, 0.159, 0.10, 0.056, 0.11, "Test", "20%", bg_color="#ffffff", border_color="#10b981", lw=1.0)

    draw_arrow(ax, 0.063, 0.28, 0.063, 0.21, color="#10b981", lw=1.2)
    draw_arrow(ax, 0.125, 0.28, 0.125, 0.21, color="#10b981", lw=1.2)
    draw_arrow(ax, 0.187, 0.28, 0.187, 0.21, color="#10b981", lw=1.2)

    # =========================================================================
    # (B) THREE MODELING STREAMS (Middle Column)
    # =========================================================================
    y_s1 = 0.69  # Classical ML
    y_s2 = 0.39  # Deep Learning Core (Layer Blocks)
    y_s3 = 0.11  # Transformer

    # Main Bold Pipeline Feeder from Train Set (x=0.215, y=0.385)
    ax.plot([0.215, 0.235], [0.385, 0.385], color="#2563eb", lw=2.2, zorder=4)
    ax.plot([0.235, 0.235], [y_s3 + 0.08, y_s1 + 0.08], color="#2563eb", lw=2.2, zorder=4)

    # Branching arrows
    draw_arrow(ax, 0.235, y_s1 + 0.08, 0.260, y_s1 + 0.08, color="#0284c7", lw=2.0)
    draw_arrow(ax, 0.235, y_s2 + 0.08, 0.260, y_s2 + 0.08, color="#d97706", lw=2.0)
    draw_arrow(ax, 0.235, y_s3 + 0.08, 0.260, y_s3 + 0.08, color="#059669", lw=2.0)

    # -------------------------------------------------------------------------
    # STREAM 1: CLASSICAL ML (Sparse TF-IDF + Linear SVM)
    # -------------------------------------------------------------------------
    draw_box(ax, 0.260, y_s1, 0.200, 0.16,
             "TF-IDF Vectorizer",
             "Word & Char N-Grams (1-2)",
             tag="V = 9,864",
             bg_color="#f0f9ff", border_color="#0284c7", title_color="#0369a1",
             tag_bg="#e0f2fe", tag_fg="#0284c7")

    draw_arrow(ax, 0.460, y_s1 + 0.08, 0.500, y_s1 + 0.08, color="#0284c7", lw=1.8)

    draw_box(ax, 0.500, y_s1, 0.235, 0.16,
             "Linear SVM / MNB",
             "Convex Margin / Bayes",
             tag="F1: 0.9917-1.0 | 0.03 ms",
             bg_color="#f0f9ff", border_color="#0284c7", title_color="#0369a1",
             tag_bg="#e0f2fe", tag_fg="#0284c7")

    # -------------------------------------------------------------------------
    # STREAM 2: PROPOSED DEEP NEURAL CORE (Layer-by-Layer Graphical Blocks)
    # -------------------------------------------------------------------------
    # Block 1: Embedding Layer
    draw_box(ax, 0.260, y_s2, 0.100, 0.16,
             "Embedding",
             "Dense Matrix",
             tag="d = 128",
             bg_color="#fffbeb", border_color="#d97706", title_color="#b45309",
             tag_bg="#fef3c7", tag_fg="#b45309")

    draw_arrow(ax, 0.360, y_s2 + 0.08, 0.380, y_s2 + 0.08, color="#d97706", lw=1.6)

    # Block 2: 1D CNN
    draw_box(ax, 0.380, y_s2, 0.100, 0.16,
             "1D CNN",
             "k ∈ {2, 3, 4}",
             tag="64 Filters",
             bg_color="#fffbeb", border_color="#d97706", title_color="#b45309",
             tag_bg="#fef3c7", tag_fg="#b45309")

    draw_arrow(ax, 0.480, y_s2 + 0.08, 0.500, y_s2 + 0.08, color="#d97706", lw=1.6)

    # Block 3: BiLSTM
    draw_box(ax, 0.500, y_s2, 0.105, 0.16,
             "BiLSTM",
             "Forward + Backward",
             tag="128 Hidden",
             bg_color="#fffbeb", border_color="#d97706", title_color="#b45309",
             tag_bg="#fef3c7", tag_fg="#b45309")

    draw_arrow(ax, 0.605, y_s2 + 0.08, 0.625, y_s2 + 0.08, color="#d97706", lw=1.6)

    # Block 4: Self-Attention & Concat
    draw_box(ax, 0.625, y_s2, 0.110, 0.16,
             "CNN-BiLSTM",
             "Spatial + Recurrent",
             tag="F1: 0.9959 | 1.64 ms",
             bg_color="#fffbeb", border_color="#d97706", title_color="#b45309",
             tag_bg="#fef3c7", tag_fg="#b45309")

    # -------------------------------------------------------------------------
    # STREAM 3: PRETRAINED TRANSFORMER (mBERT)
    # -------------------------------------------------------------------------
    draw_box(ax, 0.260, y_s3, 0.170, 0.16,
             "WordPiece Tokenizer",
             "119.5k Subword Vocab",
             tag="L = 128",
             bg_color="#ecfdf5", border_color="#059669", title_color="#047857",
             tag_bg="#d1fae5", tag_fg="#059669")

    draw_arrow(ax, 0.430, y_s3 + 0.08, 0.470, y_s3 + 0.08, color="#059669", lw=1.8)

    draw_box(ax, 0.470, y_s3, 0.265, 0.16,
             "Fine-Tuned mBERT Transformer",
             "12 Layers · 12 Attention Heads · 768d",
             tag="F1: 0.9836 | 108 ms",
             bg_color="#ecfdf5", border_color="#059669", title_color="#047857",
             tag_bg="#d1fae5", tag_fg="#059669")

    # =========================================================================
    # (C) CONVERGENCE & CLASSIFICATION HEAD (Right Column)
    # =========================================================================
    # Convergence collector bus
    ax.plot([0.735, 0.755], [y_s1 + 0.08, y_s1 + 0.08], color="#6b21a8", lw=2.0, zorder=4)
    ax.plot([0.735, 0.755], [y_s2 + 0.08, y_s2 + 0.08], color="#6b21a8", lw=2.0, zorder=4)
    ax.plot([0.735, 0.755], [y_s3 + 0.08, y_s3 + 0.08], color="#6b21a8", lw=2.0, zorder=4)
    ax.plot([0.755, 0.755], [y_s3 + 0.08, y_s1 + 0.08], color="#6b21a8", lw=2.0, zorder=4)

    # Feeder arrow to Classification Head
    draw_arrow(ax, 0.755, 0.58, 0.795, 0.58, color="#6b21a8", lw=2.2)

    # 1. Classification Projection Head
    draw_box(ax, 0.795, 0.48, 0.170, 0.20,
             "Dense Head & Softmax",
             "Projection: d → 2\nProbability: P(y | x)",
             tag="Softmax",
             bg_color="#ffffff", border_color="#6b21a8", title_color="#4c1d95",
             tag_bg="#f3e8ff", tag_fg="#6b21a8")

    # Downward arrow to Prediction Output
    draw_arrow(ax, 0.880, 0.48, 0.880, 0.35, color="#6b21a8", lw=2.0)

    # 2. Binary Prediction Output
    draw_box(ax, 0.795, 0.12, 0.170, 0.23,
             "Prediction Output",
             "Class 0: Non-Offensive\nClass 1: Offensive",
             tag="Argmax Decision",
             bg_color="#faf5ff", border_color="#7c3aed", title_color="#5b21b6",
             tag_bg="#ede9fe", tag_fg="#7c3aed")

    plt.tight_layout()
    png_path = os.path.join(PLOTS_DIR, "system_architecture.png")
    pdf_path = os.path.join(PLOTS_DIR, "system_architecture.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Q1 journal architecture diagrams saved to:\n  - {png_path}\n  - {pdf_path}")


if __name__ == "__main__":
    generate_q1_architecture()
