# Multilingual Offensive Language Benchmark (MOLB)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target: JIIS](https://img.shields.io/badge/Target-Journal%20of%20Intelligent%20Information%20Systems-orange.svg)](https://www.springer.com/journal/10844)

An empirically audited, leakage-controlled multi-script offensive language detection benchmark spanning four South Asian languages and three distinct writing systems:
- **Urdu** (Perso-Arabic script)
- **Pashto** (Arabic-derived Pashto script)
- **Roman Urdu** (Latin transliteration)
- **English** (Latin script)

---

## 1. Benchmark Overview & Forensic Deduplication

Prevailing social media benchmarks in low-resource and multilingual NLP frequently suffer from artificial template inflation and cross-partition data contamination. Through systematic forensic text auditing, this repository collapses a nominal corpus of **144,265 uncurated records** into a certified, partition-independent evaluation benchmark of **1,601 canonical independence groups**.

```
144,265 Nominal Raw Records (Uncurated social media scrapes with repeated templates)
   │
   ▼ Stage 1: Exact Raw String Deduplication
 17,559 Unique Raw Instances (87.83% exact redundancy reduction)
   │
   ▼ Stage 2: Normalized Clean-Text Deduplication (URL, mention, whitespace normalization)
 11,789 Unique Clean Texts (91.83% cumulative reduction)
   │
   ▼ Stage 3: Canonical Base-Template & Independence Group Consolidation
  1,601 Canonical Independence Groups (Certified zero-overlap evaluation benchmark)
```

### Partition Distribution (Canonical Benchmark)

| Split | Partition Count | Percentage | Offensive (`1`) | Non-Offensive (`0`) | Cross-Split Duplicate Overlap |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | **1,121** | 70.02% | 838 (74.76%) | 283 (25.24%) | **0.00%** |
| **Validation** | **160** | 9.99% | 120 (75.00%) | 40 (25.00%) | **0.00%** |
| **Test** | **320** | 19.99% | 238 (74.38%) | 82 (25.62%) | **0.00%** |
| **Total Benchmark** | **1,601** | 100.00% | **1,196** (74.70%) | **405** (25.30%) | **0.00% (5 representations audited)** |

*Audited representations with certified zero overlap: raw text, clean text, deduplication-normalized text, canonical base message, and independence group ID.*

---

## 2. Directory Structure

The repository is structured to separate the canonical benchmark, the audit trail, and raw records:

```
Dataset-of-Offensive-Language-/
├── Dataset/
│   ├── canonical/                    # PRIMARY EVALUATION BENCHMARK (1,601 groups)
│   │   ├── full_data.csv             # Complete canonical benchmark (N = 1,601)
│   │   ├── train.csv                 # Official training partition (N = 1,121)
│   │   ├── validation.csv            # Official validation partition (N = 160)
│   │   └── test.csv                  # Certified held-out test partition (N = 320)
│   │
│   ├── audit/                        # FORENSIC AUDIT TRAIL
│   │   ├── stage1_17559.csv          # Exact-duplicate deduplicated corpus (N = 17,559)
│   │   ├── stage2_11789.csv          # Clean-text normalized corpus (N = 11,789)
│   │   ├── removed_variants.csv      # Intra-group template variants collapsed (N = 10,188)
│   │   └── source_provenance.csv     # Origin & observed text variations for all 12 sub-corpora
│   │
│   ├── raw_or_original/              # UNCURATED NOMINAL RECORDS (Auditing only)
│   │   ├── original_144265.csv       # Unfiltered nominal scraped corpus (N = 144,265)
│   │   └── README_ORIGINAL.md        # Cautionary usage and leakage warning
│   │
│   ├── full_data.csv                 # Convenience mirror of canonical/full_data.csv
│   ├── train.csv                     # Convenience mirror of canonical/train.csv
│   ├── validation.csv                # Convenience mirror of canonical/validation.csv
│   └── test.csv                      # Convenience mirror of canonical/test.csv
│
├── experiments/                      # Experiment runners and evaluation pipelines
├── paper/                            # LaTeX manuscript, BibTeX database, and compiled PDF
├── plots/                            # High-resolution publication figures (PDF & PNG)
├── results/                          # Metric summaries, predictions, and test logs
├── data_statement.md                 # Linguistic data statement and annotation guidelines
├── SOURCE_PROVENANCE_REPORT.md       # Comprehensive provenance and duplication report
└── README.md                         # Repository documentation (this file)
```

---

## 3. Empirical Model Benchmarks

Evaluation of nine model architectures across three distinct computational paradigms on the certified held-out test split ($N = 320$):

| Architectural Paradigm | Model Architecture | Accuracy | Macro-$F_1$ | Weighted-$F_1$ | Single-Thread CPU Latency | Trainable Params / Features |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Classical ML** | **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | 0.0485 ms | 2,664 (sparse) |
| **Classical ML** | **Random Forest** | **1.0000** | **1.0000** | **1.0000** | 0.1080 ms | 9,864 (sparse) |
| **Classical ML** | **Linear SVM (Standard)** | 0.9938 | 0.9917 | 0.9937 | **0.0308 ms** | 1,333 (sparse) |
| **Classical ML** | **Logistic Regression** | 0.9938 | 0.9917 | 0.9937 | 0.0333 ms | 1,333 (sparse) |
| **Deep Learning** | **CNN-BiLSTM (Hybrid)** | 0.9969 | **0.9959** | 0.9969 | 1.6403 ms | 143,170 |
| **Deep Learning** | **TextCNN** | 0.9938 | 0.9917 | 0.9937 | 0.2970 ms | 126,018 |
| **Deep Learning** | **TextBiLSTM** | 0.9938 | 0.9917 | 0.9937 | 1.1664 ms | 151,298 |
| **Deep Learning** | **Attention-BiLSTM** | 0.9875 | 0.9836 | 0.9875 | 2.3464 ms | 151,426 |
| **Transformer** | **mBERT (`bert-base-multilingual-cased`)** | 0.9875 | 0.9836 | 0.9875 | 108.0169 ms | 110,000,000 |

### Key Empirical Findings:
1. **Pareto Dominance**: Classical lexical models (Linear SVM, Multinomial Naive Bayes) achieve top-tier performance while executing up to **3,500$\times$ faster** than fine-tuned mBERT.
2. **Template Leakage Impact**: Evaluating on a naive random split (96.95% template sharing) artificially inflates Pashto Macro-$F_1$ from **0.9522 to 1.0000** ($\Delta F_1 = +0.0478$), proving that naive splitting masks generalization deficits.
3. **Statistical Parity Testing**: Paired bootstrap tests ($B = 10,000$) and exact two-sided binomial tests show that differences between top-performing classical and deep models are **not statistically significant** ($p > 0.05$).
4. **Out-of-Domain Resilience**: Leave-One-Source-Out (LOSO) evaluations with matched in-domain testing demonstrate that Linear SVM retains **97.8% of its in-domain performance** under source domain shifts.

---

## 4. Reproducing Experiments

### Environment Setup
```bash
git clone https://github.com/AliAzam12/Dataset-of-Offensive-Language-.git
cd Dataset-of-Offensive-Language-
pip install -r requirements.txt  # numpy, scipy, pandas, scikit-learn, matplotlib, seaborn, torch
```

### Running Benchmark Baselines
```bash
# Evaluate all classical machine learning models (MNB, RF, SVM, LR)
python evaluate_all.py

# Run systematic feature ablation study
python run_ablations.py

# Run matched Leave-One-Source-Out (LOSO) cross-source evaluation
python run_source_holdout.py

# Run multi-seed repeatability experiments
python run_repeatability.py
```

---

## 5. Provenance & Sub-Corpora Audit Summary

| Source Identifier | Language | Script | Nominal Rows | Clean Unique | Collapse Rate | Observed Text Variation Pattern |
|:---|:---|:---|:---:|:---:|:---:|:---|
| `RU_CodeMixed_Private` | Roman Urdu | Latin | 24,584 | 2,901 | 88.2% | Prepending handles (`@admin`), appending URLs |
| `RU_Private_FB_Comments`| Roman Urdu | Latin | 24,122 | 1,972 | 91.8% | Seed comment permutations across mention tags |
| `RU_Social_Media_Mix` | Roman Urdu | Latin | 24,294 | 1,459 | 94.0% | Public comments with filler tokens and hashtags |
| `PS_LowResource_Private`| Pashto | Arabic-Pashto | 11,265 | 999 | 91.1% | Template slot-filled sentences with English nouns |
| `PS_Private_Social_Comments`| Pashto | Arabic-Pashto | 11,651 | 712 | 93.9% | Native comments replicated across placeholder handles |
| `PS_Public_Forum_Mix` | Pashto | Arabic-Pashto | 11,484 | 506 | 95.6% | Public forum comments duplicated across categories |
| `EN_Offensive_Benchmark_Private`| English | Latin | 8,453 | 834 | 90.1% | Benchmark comments modified with template noise |
| `EN_Public_Comments` | English | Latin | 8,153 | 575 | 92.9% | Public comments replicated across metadata tags |
| `EN_Social_Media_Mix` | English | Latin | 8,177 | 432 | 94.7% | Social comments with handle prefixes and hashtags |
| `UR_ArabicScript_Private`| Urdu | Perso-Arabic | 3,962 | 613 | 84.5% | Native script sentences with mention/emoji permutations |
| `UR_Private_Tweets` | Urdu | Perso-Arabic | 4,124 | 448 | 89.1% | Public tweets programmatically replicated |
| `UR_Social_Media_Mix` | Urdu | Perso-Arabic | 3,996 | 338 | 91.5% | Comments replicated across uniform topic slots |
| **Total Benchmark** | **4 Languages** | **3 Scripts** | **144,265** | **11,789** | **91.83%** | **Collapsed to 1,601 canonical independence groups** |

---

## 6. Citation

If you use this benchmark, code, or experimental findings in your research, please cite:

```bibtex
@article{azam2026benchmarking,
  title={Benchmarking Multi-Script Offensive Language Detection in Low-Resource South Asian Languages: An Empirical Evaluation Across Urdu, Roman Urdu, Pashto, and English},
  author={Azam, Ali},
  journal={Journal of Intelligent Information Systems},
  year={2026},
  publisher={Springer}
}
```

---

## 7. License & Ethical Disclaimer

- The canonical benchmark and evaluation code are released under the [MIT License](LICENSE).
- All social media records have been completely anonymized: user account handles (`@user`), URLs, and personally identifiable information (PII) have been stripped.
- This dataset contains offensive language collected exclusively for the scientific development and evaluation of automated content moderation systems.
