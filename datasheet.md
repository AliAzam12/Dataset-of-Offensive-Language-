# Datasheet for the Multilingual Offensive Language Resource

This datasheet follows a dataset-card style structure to make the resource easier to inspect, reuse, and cite.

## Motivation

### For what purpose was the dataset created?

The dataset was created to support research on multilingual offensive-language detection in low-resource and code-mixed social media settings.

It is intended to provide:

- a reusable multilingual offensive-language resource
- a harmonised binary label space
- fixed train, validation, and test splits
- benchmark results across multiple model families
- a basis for comparing future systems under the same evaluation protocol

### Who created the dataset?

Muhammad Ali Azam Khattak.

### Who funded the dataset?

No funding was received for this study.

## Composition

### What does each instance represent?

Each instance represents a social media text sample labelled for offensive-language detection.

### What languages are included?

- Urdu
- Roman Urdu
- Pashto
- English

### How many instances are included?

144,265 samples.

### What labels are included?

- `offensive`
- `non-offensive`

### What files are included?

- `full data.csv`
- `train data.csv`
- `validation data.csv`
- `test data.csv`
- `dataset_summary_counts.csv`
- `README_DATASET.txt`
- `CITATION.cff`

## Collection process

### How was the data collected?

The corpus was constructed by combining heterogeneous offensive-language datasets/resources for Urdu, Roman Urdu, Pashto, and English social media text. Source labels were standardised into a shared binary label space.

### Were labels newly annotated?

The dataset primarily harmonises existing source labels into a common binary benchmark. Dataset-specific ambiguous labels were reviewed and mapped according to their original annotation meaning.

### Were any records removed?

Empty or malformed records were removed or corrected during corpus standardisation.

## Preprocessing

### What preprocessing was applied?

A conservative language-aware preprocessing pipeline was used. It reduces platform noise while preserving offensive cues and language-specific information.

Typical operations include:

- removing URLs
- removing user mentions
- normalising duplicate spaces
- preserving Arabic-script characters for Urdu and Pashto
- reducing excessive repeated characters where appropriate
- preserving useful hashtags or informal markers when they carry meaning

## Recommended uses

This dataset is suitable for:

- multilingual offensive-language detection research
- low-resource NLP benchmarking
- code-mixed and Romanised text classification
- model comparison across traditional ML, deep learning, transformers, and LLM adaptation
- moderation-support research with appropriate human review

## Out-of-scope uses

This dataset should not be used for:

- harassment, profiling, or targeting individuals or groups
- generating harmful language
- commercial deployment without verifying source licences
- fully automated moderation decisions without bias testing and human oversight
- making broad claims about any language, ethnicity, nationality, or community