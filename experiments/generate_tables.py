"""
Phase 7: Main Model Comparison Table Generator.
Automatically compiles results from:
- results/baselines_results.csv (Traditional ML)
- results/deep_learning_results.csv (Deep Learning)
- results/transformer_results.csv (Transformers)
- results/llm_results.csv (LLMs / Hardware constraint audit)
Outputs:
- tables/main_model_comparison.csv
Does not manually hardcode any values.
"""

import os
import sys
import logging
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_main_comparison_table():
    results_dir = os.path.join(ROOT_DIR, "results")
    tables_dir = os.path.join(ROOT_DIR, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    dfs = []

    # 1. Baselines
    bl_path = os.path.join(results_dir, "baselines_results.csv")
    if os.path.exists(bl_path):
        df_bl = pd.read_csv(bl_path)
        for _, row in df_bl.iterrows():
            dfs.append({
                "Family": row["family"],
                "Model": row["model"],
                "Accuracy": round(float(row["accuracy"]), 4),
                "Precision": round(float(row["macro_precision"]), 4),
                "Recall": round(float(row["macro_recall"]), 4),
                "Macro-F1": round(float(row["macro_f1"]), 4),
                "Weighted-F1": round(float(row["weighted_f1"]), 4),
                "Training Time (s)": round(float(row["train_time_sec"]), 2),
                "Inference Latency (ms)": round(float(row["infer_latency_ms"]), 4)
            })

    # 2. Deep Learning
    dl_path = os.path.join(results_dir, "deep_learning_results.csv")
    if os.path.exists(dl_path):
        df_dl = pd.read_csv(dl_path)
        for _, row in df_dl.iterrows():
            dfs.append({
                "Family": row["family"],
                "Model": row["model"],
                "Accuracy": round(float(row["accuracy"]), 4),
                "Precision": round(float(row["macro_precision"]), 4),
                "Recall": round(float(row["macro_recall"]), 4),
                "Macro-F1": round(float(row["macro_f1"]), 4),
                "Weighted-F1": round(float(row["weighted_f1"]), 4),
                "Training Time (s)": round(float(row["train_time_sec"]), 2),
                "Inference Latency (ms)": round(float(row["infer_latency_ms"]), 4)
            })

    # 3. Transformers
    tf_path = os.path.join(results_dir, "transformer_results.csv")
    if os.path.exists(tf_path):
        df_tf = pd.read_csv(tf_path)
        for _, row in df_tf.iterrows():
            if row.get("status") == "completed" and pd.notna(row.get("macro_f1")):
                dfs.append({
                    "Family": row["family"],
                    "Model": row["model"],
                    "Accuracy": round(float(row["accuracy"]), 4),
                    "Precision": round(float(row["macro_precision"]), 4),
                    "Recall": round(float(row["macro_recall"]), 4),
                    "Macro-F1": round(float(row["macro_f1"]), 4),
                    "Weighted-F1": round(float(row["weighted_f1"]), 4),
                    "Training Time (s)": round(float(row["train_time_sec"]), 2),
                    "Inference Latency (ms)": round(float(row["infer_latency_ms"]), 4)
                })

    main_df = pd.DataFrame(dfs)
    # Sort descending by Macro-F1
    main_df = main_df.sort_values(by="Macro-F1", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(tables_dir, "main_model_comparison.csv")
    main_df.to_csv(out_csv, index=False)
    logging.info(f"Saved main model comparison table to {out_csv}")
    print(main_df.to_string(index=False))
    return main_df


if __name__ == "__main__":
    generate_main_comparison_table()
