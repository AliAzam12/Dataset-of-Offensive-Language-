# Multilingual Offensive Language Resource and Benchmark

This repository contains a multilingual offensive-language resource and benchmark for **Urdu, Roman Urdu, Pashto, and English** social media text.

The resource is designed for research on low-resource, multilingual, Romanised, and code-mixed offensive-language detection. It provides a harmonised binary label space, fixed benchmark splits, summary counts, documentation, citation metadata, and reproducibility scripts.

## Dataset summary

| Item | Description |
|---|---|
| Task | Binary offensive-language classification |
| Languages | Urdu, Roman Urdu, Pashto, English |
| Total samples | 144,265 |
| Labels | `offensive`, `non-offensive` |
| Splits | Train, validation, test |
| Primary use | Language resource research, benchmark evaluation, moderation-support experiments |

## Language distribution

| Language | Samples |
|---|---:|
| Roman Urdu | 73,000 |
| Pashto | 34,400 |
| English | 24,783 |
| Urdu | 12,082 |
| **Total** | **144,265** |

## Label distribution

| Label | Samples |
|---|---:|
| Offensive | 85,161 |
| Non-offensive | 59,104 |
| **Total** | **144,265** |

## Repository structure

```text
.
├── README.md
├── README_DATASET.txt
├── CITATION.cff
├── LICENSE
├── datasheet.md
├── data_statement.md
├── label_mapping.md
├── ethical_use.md
├── source_provenance_template.csv
├── dataset_summary_counts.csv
├── full data.csv
├── train data.csv
├── validation data.csv
├── test data.csv
├── requirements.txt
├── model_configs.yml
└── scripts/
    ├── preprocessing.py
    ├── evaluate_baselines.py
    ├── check_dataset.py
    └── create_summary_counts.py
```

## Data files

| File | Description |
|---|---|
| `full data.csv` | Complete multilingual dataset |
| `train data.csv` | Training split |
| `validation data.csv` | Validation split |
| `test data.csv` | Test split |
| `dataset_summary_counts.csv` | Language-level and label-level summary counts |
| `README_DATASET.txt` | Dataset description and column information |
| `CITATION.cff` | Citation metadata |

## Expected columns

The released CSV files should contain the following columns where available:

| Column | Description |
|---|---|
| `id` | Unique record identifier |
| `text` | Original text sample |
| `clean_text` | Cleaned text after preprocessing |
| `language` | Urdu, Roman Urdu, Pashto, or English |
| `script` | Writing script or romanisation category |
| `source_platform` | Source platform/category metadata |
| `source_dataset` | Source dataset or source category |
| `original_label` | Label before harmonisation |
| `final_label` | Harmonised label: offensive or non-offensive |
| `label_id` | Numeric label ID |
| `split` | Train, validation, or test |
| `topic` | Optional topic/category metadata |
| `metadata_flags` | Optional indicators such as URL, emoji, hashtag, length, token count, year |

Column names may be adapted by users, but benchmark reproduction should preserve the same train/validation/test split files.

## Label space

All original source labels were harmonised into a binary label space:

| Original/source label family | Harmonised label |
|---|---|
| hate speech, abusive, toxic, offensive | offensive |
| insult, obscene, threat, targeted abuse | offensive |
| neutral, clean, normal, non-toxic | non-offensive |
| dataset-specific ambiguous labels | reviewed and mapped according to original annotation meaning |

See [`label_mapping.md`](label_mapping.md) for details.

## Quick start

Install the requirements:

```bash
pip install -r requirements.txt
```

Check dataset structure:

```bash
python scripts/check_dataset.py --data "full data.csv"
```

Run preprocessing and create a cleaned copy:

```bash
python scripts/preprocessing.py --input "full data.csv" --output "full_data_cleaned.csv"
```

Evaluate a simple TF-IDF baseline:

```bash
python scripts/evaluate_baselines.py \
  --train "train data.csv" \
  --valid "validation data.csv" \
  --test "test data.csv" \
  --text-column clean_text \
  --label-column final_label
```

Create summary counts:

```bash
python scripts/create_summary_counts.py --data "full data.csv" --output "dataset_summary_counts_generated.csv"
```

## Benchmark protocol

To make future results comparable, use the predefined split files:

- `train data.csv`
- `validation data.csv`
- `test data.csv`

Do not create a new random split when reporting benchmark results against this repository unless you clearly state that your setting is not directly comparable with the official benchmark.

Recommended metrics:

- Accuracy
- Precision
- Recall
- Macro-F1

Macro-F1 is recommended as the primary metric because the dataset contains class imbalance and multilingual variation.

## Responsible use

This dataset contains offensive and potentially harmful language. It is released for research, education, benchmark evaluation, and moderation-support experiments only. Users should avoid exposing annotators, students, or downstream users to harmful text unnecessarily and should not use the dataset to target individuals or communities.

See [`ethical_use.md`](ethical_use.md).

## Citation

Please cite the associated paper and the repository metadata when using this dataset. See [`CITATION.cff`](CITATION.cff).

## Important note on source provenance

This repository may contain data derived from multiple public or research datasets. Users should confirm the original dataset licences and platform terms before redistributing, modifying, or using the data for commercial purposes. The file [`source_provenance_template.csv`](source_provenance_template.csv) is included to help document source names, URLs, licences, and permissions.
