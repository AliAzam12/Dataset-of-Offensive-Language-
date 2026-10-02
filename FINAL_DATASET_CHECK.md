# Final Scientific Dataset Integrity & Provenance Check

**Target Venue**: *Journal of Intelligent Information Systems* (Springer)  
**Document**: `FINAL_DATASET_CHECK.md`  
**Status**: Pre-Modeling Final Certification  
**Random Seed**: 42  

---

## 1. Executive Summary

This audit and repair resolves the remaining template/near-duplicate leakage in the multilingual offensive language benchmark.

1. **Root Cause Resolved**: The previous 11,789-record dataset contained widespread artificial template inflation. By detecting superficial variations (mentions, emojis, hashtags, URLs, punctuation, extra whitespace, and minor word-order permutations), all rows were clustered into **1,601 unique independence groups** (`independence_group_id`).
2. **Canonical Consolidation**: Exactly **one canonical observation** per independence group was retained in the main benchmark (`full_data_final.csv`), reducing the dataset to **1,601 genuinely independent records**.
3. **Audit Trail**: All **10,188 removed template variants** are preserved in [`removed_variants.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/removed_variants.csv) with their parent canonical mapping, full text variants, and original row IDs.
4. **Absolute Zero Leakage**: All five representations (`text`, `clean_text`, `dedup_normalized_text`, `base_message`, and `independence_group_id`) have **0 overlap** between Train, Validation, and Test.

---

## 2. Quantitative Summary

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **Original Nominal Rows** | **144,265** | 100.0% |
| **Exact Deduplicated Rows (Step 1)** | **17,559** | 12.17% |
| **Clean-Text Deduplicated Rows (Step 2)** | **11,789** | 8.17% |
| **Final Independent Canonical Rows** | **1,601** | **1.11%** |
| **Removed Template Variants** | **10,188** | 86.42% of 11.7k set |
| **Total Original Duplicate Rows Collapsed** | **142,664** | **98.89%** |

---

## 3. Critical Zero-Leakage Verification (All 5 Checks = 0)

All split overlap assertions were evaluated across **Train (1,121)**, **Validation (160)**, and **Test (320)**.

| Representation | Check | Overlap Count | Status |
| :--- | :--- | :---: | :---: |
| **1. Raw Text (`text`)** | `train_text ∩ val_text` | **0** | PASSED |
| | `train_text ∩ test_text` | **0** | PASSED |
| | `val_text ∩ test_text` | **0** | PASSED |
| **2. Preprocessed Text (`clean_text`)** | `train_clean_text ∩ val_clean_text` | **0** | PASSED |
| | `train_clean_text ∩ test_clean_text` | **0** | PASSED |
| | `val_clean_text ∩ test_clean_text` | **0** | PASSED |
| **3. Conservative Normalized (`dedup_normalized_text`)** | `train_dedup_norm ∩ val_dedup_norm` | **0** | PASSED |
| | `train_dedup_norm ∩ test_dedup_norm` | **0** | PASSED |
| | `val_dedup_norm ∩ test_dedup_norm` | **0** | PASSED |
| **4. Base Message (`base_message`)** | `train_base_message ∩ val_base_message` | **0** | PASSED |
| | `train_base_message ∩ test_base_message` | **0** | PASSED |
| | `val_base_message ∩ test_base_message` | **0** | PASSED |
| **5. Independence Group (`independence_group_id`)** | `train_ind_group ∩ val_ind_group` | **0** | PASSED |
| | `train_ind_group ∩ test_ind_group` | **0** | PASSED |
| | `val_ind_group ∩ test_ind_group` | **0** | PASSED |

**Verdict**: The final train, validation, and test splits have **ZERO template or exact leakage**. A model trained on `train_final.csv` cannot memorize test sentence patterns.

---

## 4. Final Split & Demographic Distributions

### Split Sizes
- **Train (`Dataset/train.csv`)**: **1,121** rows (**70.02%**)
- **Validation (`Dataset/validation.csv`)**: **160** rows (**9.99%**)
- **Test (`Dataset/test.csv`)**: **320** rows (**19.99%**)
- **Total (`Dataset/full_data.csv`)**: **1,601** rows (**100.00%**)

### By Language
| Language | Script | Total Samples | Train | Validation | Test | Train % | Test % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Roman Urdu** | Latin | **822** | 576 | 82 | 164 | 70.07% | 19.95% |
| **English** | Latin | **321** | 225 | 32 | 64 | 70.09% | 19.94% |
| **Urdu** | Arabic-Urdu | **243** | 170 | 24 | 49 | 69.96% | 20.16% |
| **Pashto** | Arabic-Pashto | **215** | 150 | 22 | 43 | 69.77% | 20.00% |
| **Corpus Total** | — | **1,601** | **1,121** | **160** | **320** | **70.02%** | **19.99%** |

### By Label
| Final Label | Binary ID | Total Samples | Percentage | Train | Validation | Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Offensive** | `1` | **1,196** | 74.70% | 837 | 120 | 239 |
| **Non-Offensive** | `0` | **405** | 25.30% | 284 | 40 | 81 |

---

## 5. Source Provenance & Integrity Concerns

### Source Dataset Breakdown in Final Benchmark
| Source Dataset | Language | Canonical Records | Cross-Source Overlap | Original Rows Claimed | Integrity Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`RU_CodeMixed_Private`** | Roman Urdu | 461 | 435 | 24,584 | `REQUIRES HUMAN VERIFICATION` |
| **`RU_Private_FB_Comments`** | Roman Urdu | 224 | 207 | 24,122 | `REQUIRES HUMAN VERIFICATION` |
| **`EN_Offensive_Benchmark_Private`** | English | 160 | 152 | 8,453 | `REQUIRES HUMAN VERIFICATION` |
| **`RU_Social_Media_Mix`** | Roman Urdu | 137 | 131 | 24,294 | `REQUIRES HUMAN VERIFICATION` |
| **`UR_ArabicScript_Private`** | Urdu | 129 | 119 | 3,962 | `REQUIRES HUMAN VERIFICATION` |
| **`PS_LowResource_Private`** | Pashto | 121 | 108 | 11,265 | `REQUIRES HUMAN VERIFICATION` |
| **`EN_Public_Comments`** | English | 103 | 91 | 8,153 | `REQUIRES HUMAN VERIFICATION` |
| **`PS_Private_Social_Comments`** | Pashto | 73 | 64 | 11,651 | `REQUIRES HUMAN VERIFICATION` |
| **`UR_Private_Tweets`** | Urdu | 70 | 66 | 4,124 | `REQUIRES HUMAN VERIFICATION` |
| **`EN_Social_Media_Mix`** | English | 58 | 51 | 8,177 | `REQUIRES HUMAN VERIFICATION` |
| **`UR_Social_Media_Mix`** | Urdu | 44 | 41 | 3,996 | `REQUIRES HUMAN VERIFICATION` |
| **`PS_Public_Forum_Mix`** | Pashto | 21 | 18 | 11,484 | `REQUIRES HUMAN VERIFICATION` |

### Key Provenance Alerts:
1. **Unverifiable Institutional Provenance**: All 12 datasets—especially the 6 containing `Private`—lack scraping logs, author attributions, original URLs, and licensing documentation in the workspace. They are flagged as **`REQUIRES HUMAN VERIFICATION`**.
2. **Heavy Cross-Source Duplication**: **1,415 of the 1,601 base messages (88.38%)** were copied across multiple source dataset names in the original corpus. This proves the original source identifiers were assigned post-hoc across the same seed sentences.
3. **Synthetic Template Augmentation**: The original 144k corpus was formed by inflating ~1,601 core sentences using rotating mentions (`@admin`, `@user`, etc.), a dummy URL (`https://example.com/post`), 8 hashtags, and filler phrases (`calm down`, `update`, `please`).

---

## 6. Generated Files in [`Dataset/`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset)

1. [`Dataset/full_data.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/full_data.csv) — Complete deduplicated, leakage-free benchmark (**1,601** rows).
2. [`Dataset/train.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/train.csv) — Training partition (**1,121** rows, 70.02%).
3. [`Dataset/validation.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/validation.csv) — Validation partition (**160** rows, 9.99%).
4. [`Dataset/test.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/test.csv) — Held-out test partition (**320** rows, 19.99%).
5. [`Dataset/removed_variants.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/removed_variants.csv) — Complete audit log of all **10,188** removed template variants.
6. [`Dataset/dataset_summary_counts.csv`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/Dataset/dataset_summary_counts.csv) — Summary distribution counts.
7. [`FINAL_DATASET_CHECK.md`](file:///e:/Github%20Repos/Dataset-of-Offensive-Language-/FINAL_DATASET_CHECK.md) — Comprehensive verification report.
