"""
Training and Evaluation Pipeline for Multilingual Transformer Models.
Supports mBERT, XLM-RoBERTa, MuRIL, IndicBERT, PashtoBERT.
Implements validation-based early stopping, checkpoint selection,
evaluation across language subsets, and prediction logging.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Ensure workspace root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils.seed import seed_everything
from src.utils.hardware import get_environment_info
from src.evaluation.metrics import compute_all_metrics
from src.models.transformer_models import CHECKPOINT_REGISTRY, load_transformer_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PretokenizedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        enc = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx]
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), np.array(all_preds), np.array(all_probs), np.array(all_labels)


def run_transformer_experiment(
    model_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = CHECKPOINT_REGISTRY[model_key]
    checkpoint_id = info["checkpoint_id"]
    logging.info(f"Initializing {model_key} ({checkpoint_id}) on device {device}...")

    seed_everything(seed)
    tokenizer, model = load_transformer_model(checkpoint_id, num_labels=2, device=device)

    train_dataset = PretokenizedDataset(train_df["clean_text"], train_df["label_id"], tokenizer, max_length)
    val_dataset = PretokenizedDataset(val_df["clean_text"], val_df["label_id"], tokenizer, max_length)
    test_dataset = PretokenizedDataset(test_df["clean_text"], test_df["label_id"], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_macro_f1 = -1.0
    best_weights_path = os.path.join(ROOT_DIR, "checkpoints", "transformers", f"{model_key}_best.pt")
    os.makedirs(os.path.dirname(best_weights_path), exist_ok=True)

    patience = 2
    patience_counter = 0
    best_epoch = 1

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds, val_probs, val_labels = evaluate_model(model, val_loader, device)
        val_metrics = compute_all_metrics(val_labels, val_preds)
        val_macro_f1 = val_metrics["macro_f1"]

        logging.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            logging.info(f"  -> New best validation Macro-F1: {best_val_macro_f1:.4f}. Saved checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  -> Early stopping triggered at epoch {epoch}")
                break

    train_time_sec = round(time.time() - t0_train, 2)

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))

    # Measure test inference latency
    t0_inf = time.time()
    _, test_preds, test_probs, test_labels = evaluate_model(model, test_loader, device)
    total_inf_time = time.time() - t0_inf
    infer_latency_ms = round((total_inf_time / len(test_df)) * 1000, 4)

    test_metrics = compute_all_metrics(
        test_labels,
        test_preds,
        languages=test_df["language"].tolist()
    )

    # Save test predictions for error analysis
    preds_df = test_df.copy()
    preds_df["predicted_label_id"] = test_preds
    preds_df["predicted_label"] = ["offensive" if p == 1 else "non-offensive" for p in test_preds]
    preds_df["confidence"] = [probs[p] for probs, p in zip(test_probs, test_preds)]
    preds_df["model"] = model_key
    preds_path = os.path.join(ROOT_DIR, "results", f"preds_{model_key}.csv")
    preds_df.to_csv(preds_path, index=False)

    result_row = {
        "family": "Transformer",
        "model": model_key,
        "exact_checkpoint_id": checkpoint_id,
        "best_epoch": best_epoch,
        "val_macro_f1": round(best_val_macro_f1, 4),
        "accuracy": round(test_metrics["accuracy"], 4),
        "macro_precision": round(test_metrics["macro_precision"], 4),
        "macro_recall": round(test_metrics["macro_recall"], 4),
        "macro_f1": round(test_metrics["macro_f1"], 4),
        "weighted_f1": round(test_metrics["weighted_f1"], 4),
        "offensive_f1": round(test_metrics["offensive_f1"], 4),
        "non_offensive_f1": round(test_metrics["non_offensive_f1"], 4),
        "train_time_sec": train_time_sec,
        "infer_latency_ms": infer_latency_ms,
        "checkpoint_path": best_weights_path,
        "status": "completed"
    }

    # Add per-language metrics
    for lang, lmetrics in test_metrics.get("languages", {}).items():
        result_row[f"{lang}_macro_f1"] = round(lmetrics["macro_f1"], 4)

    return result_row


def main():
    train_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "validation.csv"))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, "Dataset", "test.csv"))
    results_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_transformer_results = []

    # 1. mBERT (bert-base-multilingual-cased)
    logging.info("==================== TRAINING mBERT ====================")
    try:
        mbert_res = run_transformer_experiment(
            model_key="mBERT",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            epochs=3,
            lr=2e-5,
            batch_size=16,
            max_length=128,
            seed=42
        )
        all_transformer_results.append(mbert_res)
    except Exception as e:
        logging.error(f"Error training mBERT: {e}", exc_info=True)

    # 2. IndicBERT (gated repository status report)
    indic_info = CHECKPOINT_REGISTRY["IndicBERT"]
    all_transformer_results.append({
        "family": "Transformer",
        "model": "IndicBERT",
        "exact_checkpoint_id": indic_info["checkpoint_id"],
        "best_epoch": None,
        "val_macro_f1": None,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_f1": None,
        "offensive_f1": None,
        "non_offensive_f1": None,
        "train_time_sec": None,
        "infer_latency_ms": None,
        "checkpoint_path": None,
        "status": f"Skipped ({indic_info['notes']})"
    })

    # 3. PashtoBERT (unverified repository status report)
    pashto_info = CHECKPOINT_REGISTRY["PashtoBERT"]
    all_transformer_results.append({
        "family": "Transformer",
        "model": "PashtoBERT",
        "exact_checkpoint_id": pashto_info["checkpoint_id"],
        "best_epoch": None,
        "val_macro_f1": None,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_f1": None,
        "offensive_f1": None,
        "non_offensive_f1": None,
        "train_time_sec": None,
        "infer_latency_ms": None,
        "checkpoint_path": None,
        "status": f"Skipped ({pashto_info['notes']})"
    })

    out_df = pd.DataFrame(all_transformer_results)
    out_csv = os.path.join(results_dir, "transformer_results.csv")
    out_df.to_csv(out_csv, index=False)
    logging.info(f"Saved transformer results to {out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

