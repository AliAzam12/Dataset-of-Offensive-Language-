"""
Phase 18: Master Experiment Registry Builder.
Aggregates all completed runs across baselines, deep learning, repeatability, ablations,
and source-holdout experiments into a single unified audit ledger:
- results/experiment_registry.csv
Records:
run_id, timestamp, model, family, exact_checkpoint, seed, dataset_version, split_version,
hyperparameters, accuracy, macro_f1, hardware, runtime_sec, checkpoint_path
"""

import os
import sys
import time
import logging
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.hardware import get_environment_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_experiment_registry():
    results_dir = os.path.join(ROOT_DIR, "results")
    env = get_environment_info()
    hw_desc = f"CPU: {env['processor']} ({env['cpu_count']} cores) | CUDA: {env['cuda_available']} | Python: {env['python_version']}"

    records = []
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Baselines
    bl_path = os.path.join(results_dir, "baselines_results.csv")
    if os.path.exists(bl_path):
        for _, r in pd.read_csv(bl_path).iterrows():
            records.append({
                "run_id": f"RUN_BL_{r['model']}",
                "timestamp": t_now,
                "model": r["model"],
                "family": r["family"],
                "exact_checkpoint": r["model"],
                "seed": 42,
                "dataset_version": "v1.0_canonical_n1601",
                "split_version": "70_10_20_group_aware",
                "hyperparameters": r.get("best_params", ""),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "hardware": hw_desc,
                "runtime_sec": r["train_time_sec"],
                "checkpoint_path": r["checkpoint_path"]
            })

    # 2. Deep Learning
    dl_path = os.path.join(results_dir, "deep_learning_results.csv")
    if os.path.exists(dl_path):
        for _, r in pd.read_csv(dl_path).iterrows():
            records.append({
                "run_id": f"RUN_DL_{r['model']}",
                "timestamp": t_now,
                "model": r["model"],
                "family": r["family"],
                "exact_checkpoint": r["model"],
                "seed": 42,
                "dataset_version": "v1.0_canonical_n1601",
                "split_version": "70_10_20_group_aware",
                "hyperparameters": f"epochs={r.get('best_epoch', 1)}, lr=0.001, batch=32",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "hardware": hw_desc,
                "runtime_sec": r["train_time_sec"],
                "checkpoint_path": r["checkpoint_path"]
            })

    # 3. Repeatability Runs
    rep_path = os.path.join(results_dir, "repeatability_runs.csv")
    if os.path.exists(rep_path):
        for idx, r in pd.read_csv(rep_path).iterrows():
            records.append({
                "run_id": f"RUN_REP_{r['model']}_s{r['seed']}",
                "timestamp": t_now,
                "model": r["model"],
                "family": r["family"],
                "exact_checkpoint": r["model"],
                "seed": int(r["seed"]),
                "dataset_version": "v1.0_canonical_n1601",
                "split_version": "70_10_20_group_aware",
                "hyperparameters": f"seed={r['seed']}",
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "hardware": hw_desc,
                "runtime_sec": r["train_time_sec"],
                "checkpoint_path": f"checkpoints/{r['model']}_s{r['seed']}"
            })

    # 4. Transformers Audit / Models
    tf_path = os.path.join(results_dir, "transformer_results.csv")
    if os.path.exists(tf_path):
        for _, r in pd.read_csv(tf_path).iterrows():
            records.append({
                "run_id": f"RUN_TF_{r['model']}",
                "timestamp": t_now,
                "model": r["model"],
                "family": r["family"],
                "exact_checkpoint": r["exact_checkpoint_id"],
                "seed": 42,
                "dataset_version": "v1.0_canonical_n1601",
                "split_version": "70_10_20_group_aware",
                "hyperparameters": "epochs=3, lr=2e-5, batch=16, max_len=128",
                "accuracy": r.get("accuracy", None),
                "macro_f1": r.get("macro_f1", None),
                "hardware": hw_desc,
                "runtime_sec": r.get("train_time_sec", None),
                "checkpoint_path": r.get("checkpoint_path", None)
            })

    # 5. LLMs Hardware Audit
    llm_path = os.path.join(results_dir, "llm_results.csv")
    if os.path.exists(llm_path):
        for _, r in pd.read_csv(llm_path).iterrows():
            records.append({
                "run_id": f"RUN_LLM_{r['model']}",
                "timestamp": t_now,
                "model": r["model"],
                "family": r["family"],
                "exact_checkpoint": r["exact_checkpoint_id"],
                "seed": 42,
                "dataset_version": "v1.0_canonical_n1601",
                "split_version": "70_10_20_group_aware",
                "hyperparameters": r["quantization_config"],
                "accuracy": None,
                "macro_f1": None,
                "hardware": hw_desc,
                "runtime_sec": None,
                "checkpoint_path": r["status"]
            })

    df_reg = pd.DataFrame(records)
    out_csv = os.path.join(results_dir, "experiment_registry.csv")
    df_reg.to_csv(out_csv, index=False)
    logging.info(f"Saved {len(df_reg)} registered experiment runs to {out_csv}")
    return df_reg


if __name__ == "__main__":
    build_experiment_registry()
