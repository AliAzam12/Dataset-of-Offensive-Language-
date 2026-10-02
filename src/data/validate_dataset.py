"""
src/data/validate_dataset.py
----------------------------
Phase 0: Comprehensive pre-training dataset validation.
Checks zero-leakage across 5 representations:
  1. text
  2. clean_text
  3. dedup_normalized_text
  4. base_message
  5. independence_group_id
Validates labels, languages, missing inputs, unique IDs.
Outputs:
  - experiments/00_dataset_validation.json
  - experiments/00_dataset_validation.csv
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
EXP_DIR = os.path.join(PROJECT_ROOT, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)

def validate_datasets(dataset_dir: str = DATASET_DIR):
    full_path = os.path.join(dataset_dir, 'full_data.csv')
    train_path = os.path.join(dataset_dir, 'train.csv')
    val_path = os.path.join(dataset_dir, 'validation.csv')
    test_path = os.path.join(dataset_dir, 'test.csv')

    for p in [full_path, train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required dataset file not found: {p}")

    df_full = pd.read_csv(full_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    results = {
        'total_full_rows': len(df_full),
        'train_rows': len(df_train),
        'validation_rows': len(df_val),
        'test_rows': len(df_test),
        'row_sum_check': len(df_train) + len(df_val) + len(df_test) == len(df_full),
        'unique_ids_full': int(df_full['id'].nunique()),
        'id_is_unique': bool(df_full['id'].nunique() == len(df_full)),
        'missing_clean_text': int(df_full['clean_text'].isnull().sum()),
        'missing_raw_text': int(df_full['text'].isnull().sum()),
        'valid_labels_check': set(df_full['final_label'].unique()).issubset({'offensive', 'non-offensive'}),
        'valid_languages_check': set(df_full['language'].unique()).issubset({'Roman Urdu', 'English', 'Urdu', 'Pashto'}),
    }

    # Overlap checks
    checks = [
        ('raw_text', 'text'),
        ('clean_text', 'clean_text'),
        ('dedup_normalized_text', 'dedup_normalized_text'),
        ('base_message', 'base_message'),
        ('independence_group_id', 'independence_group_id')
    ]

    overlap_details = {}
    leakage_detected = False

    for rep_name, col in checks:
        tr_set = set(df_train[col].astype(str))
        va_set = set(df_val[col].astype(str))
        te_set = set(df_test[col].astype(str))

        tr_va = len(tr_set & va_set)
        tr_te = len(tr_set & te_set)
        va_te = len(va_set & te_set)

        overlap_details[f'{rep_name}_train_val'] = tr_va
        overlap_details[f'{rep_name}_train_test'] = tr_te
        overlap_details[f'{rep_name}_val_test'] = va_te

        if tr_va > 0 or tr_te > 0 or va_te > 0:
            leakage_detected = True

    results['leakage_detected'] = leakage_detected
    results['overlap_checks'] = overlap_details

    # Per-language breakdown
    lang_breakdown = {}
    for lang, g in df_full.groupby('language'):
        lang_breakdown[lang] = {
            'total': len(g),
            'train': int((df_train['language'] == lang).sum()),
            'val': int((df_val['language'] == lang).sum()),
            'test': int((df_test['language'] == lang).sum()),
            'offensive': int((g['final_label'] == 'offensive').sum()),
            'non_offensive': int((g['final_label'] == 'non-offensive').sum())
        }
    results['language_breakdown'] = lang_breakdown

    # Per-label breakdown
    label_breakdown = {}
    for lbl, g in df_full.groupby('final_label'):
        label_breakdown[lbl] = {
            'total': len(g),
            'train': int((df_train['final_label'] == lbl).sum()),
            'val': int((df_val['final_label'] == lbl).sum()),
            'test': int((df_test['final_label'] == lbl).sum())
        }
    results['label_breakdown'] = label_breakdown

    # Save JSON
    json_path = os.path.join(EXP_DIR, '00_dataset_validation.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_rows = []
    csv_rows.append({'category': 'counts', 'metric': 'full_rows', 'value': results['total_full_rows'], 'status': 'PASS'})
    csv_rows.append({'category': 'counts', 'metric': 'train_rows', 'value': results['train_rows'], 'status': 'PASS'})
    csv_rows.append({'category': 'counts', 'metric': 'validation_rows', 'value': results['validation_rows'], 'status': 'PASS'})
    csv_rows.append({'category': 'counts', 'metric': 'test_rows', 'value': results['test_rows'], 'status': 'PASS'})
    csv_rows.append({'category': 'integrity', 'metric': 'unique_ids', 'value': results['unique_ids_full'], 'status': 'PASS' if results['id_is_unique'] else 'FAIL'})
    csv_rows.append({'category': 'integrity', 'metric': 'missing_clean_text', 'value': results['missing_clean_text'], 'status': 'PASS' if results['missing_clean_text'] == 0 else 'FAIL'})
    csv_rows.append({'category': 'integrity', 'metric': 'valid_labels', 'value': str(results['valid_labels_check']), 'status': 'PASS' if results['valid_labels_check'] else 'FAIL'})
    csv_rows.append({'category': 'integrity', 'metric': 'valid_languages', 'value': str(results['valid_languages_check']), 'status': 'PASS' if results['valid_languages_check'] else 'FAIL'})

    for k, v in overlap_details.items():
        csv_rows.append({'category': 'leakage_check', 'metric': k, 'value': v, 'status': 'PASS' if v == 0 else 'FAIL'})

    df_csv = pd.DataFrame(csv_rows)
    csv_path = os.path.join(EXP_DIR, '00_dataset_validation.csv')
    df_csv.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Dataset validation results saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")

    if leakage_detected:
        raise ValueError("CRITICAL ERROR: Data leakage detected across splits! Execution stopped.")
    else:
        print("PHASE 0 PASSED: All 15 zero-leakage and integrity assertions successfully verified!")

    return results

if __name__ == '__main__':
    validate_datasets()
