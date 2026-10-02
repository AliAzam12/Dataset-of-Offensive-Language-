"""
src/evaluation/metrics.py
-------------------------
Common Evaluation Pipeline for all models.
Primary metric: Macro-F1
Additional metrics:
  - Accuracy, Precision, Recall, Weighted-F1
  - Class-wise F1: Offensive (1), Non-Offensive (0)
  - Confusion Matrix
  - Stratified per-language evaluations (Urdu, Roman Urdu, Pashto, English)
Outputs: JSON + CSV per run
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix
)

def compute_metrics(y_true, y_pred, languages=None, model_name="model", run_id=None, save_dir=None):
    """
    Computes global and language-conditioned evaluation metrics.
    Ensures binary format: 1 (offensive), 0 (non-offensive).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Global metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Class-wise F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    non_off_f1 = float(f1_per_class[0])
    off_f1 = float(f1_per_class[1])

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    result = {
        'run_id': run_id,
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'macro_precision': round(float(p_macro), 4),
        'macro_recall': round(float(r_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'weighted_f1': round(float(f1_weighted), 4),
        'offensive_f1': round(off_f1, 4),
        'non_offensive_f1': round(non_off_f1, 4),
        'confusion_matrix': cm,
        'languages': {}
    }

    # Language-conditioned metrics
    if languages is not None:
        lang_array = np.array(languages)
        unique_langs = sorted(list(set(lang_array)))
        for lang in unique_langs:
            mask = (lang_array == lang)
            if np.sum(mask) == 0:
                continue
            yt_lang = y_true[mask]
            yp_lang = y_pred[mask]

            l_acc = float(accuracy_score(yt_lang, yp_lang))
            _, _, l_f1_macro, _ = precision_recall_fscore_support(yt_lang, yp_lang, average='macro', zero_division=0)
            _, _, l_f1_class, _ = precision_recall_fscore_support(yt_lang, yp_lang, average=None, labels=[0, 1], zero_division=0)

            result['languages'][lang] = {
                'samples': int(np.sum(mask)),
                'accuracy': round(l_acc, 4),
                'macro_f1': round(float(l_f1_macro), 4),
                'offensive_f1': round(float(l_f1_class[1]), 4),
                'non_offensive_f1': round(float(l_f1_class[0]), 4),
                'confusion_matrix': confusion_matrix(yt_lang, yp_lang, labels=[0, 1]).tolist()
            }

    # Save to disk if save_dir provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = f"{run_id}_{model_name}".replace('/', '_')
        json_path = os.path.join(save_dir, f"{base_name}_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Flat CSV record
        csv_record = {
            'run_id': run_id,
            'model_name': model_name,
            'accuracy': result['accuracy'],
            'macro_precision': result['macro_precision'],
            'macro_recall': result['macro_recall'],
            'macro_f1': result['macro_f1'],
            'weighted_f1': result['weighted_f1'],
            'offensive_f1': result['offensive_f1'],
            'non_offensive_f1': result['non_offensive_f1']
        }
        for lang, ldict in result['languages'].items():
            csv_record[f"{lang}_samples"] = ldict['samples']
            csv_record[f"{lang}_accuracy"] = ldict['accuracy']
            csv_record[f"{lang}_macro_f1"] = ldict['macro_f1']

        df_csv = pd.DataFrame([csv_record])
        csv_path = os.path.join(save_dir, f"{base_name}_metrics.csv")
        df_csv.to_csv(csv_path, index=False, encoding='utf-8')

    return result


compute_all_metrics = compute_metrics
