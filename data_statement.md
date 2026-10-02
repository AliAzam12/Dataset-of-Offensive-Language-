# Data Statement for the Multilingual Offensive Language Benchmark (MOLB)

**Following the Data Statements for Natural Language Processing framework (Bender and Friedman, 2018).**

---

## A. Curation Rationale
Automated offensive language detection is crucial for online community moderation, but benchmarks across low-resource South Asian languages often suffer from subtle train-test data contamination, artificial template inflation, and lack of cross-partition independence. 

The Multilingual Offensive Language Benchmark (MOLB) was curated to provide a certified, leakage-controlled multi-script benchmark across four languages and three scripts: **Urdu (Perso-Arabic), Pashto (Arabic-Pashto), Roman Urdu (Latin transliteration), and English (Latin)**. A nominal scraped corpus of 144,265 records characterized by widespread template variations was forensically audited, deduplicated, and consolidated into **1,601 canonical independence groups**, partitioned under a strict group-aware stratified protocol (1,121 train, 160 validation, and 320 test samples).

---

## B. Language Varieties
The benchmark covers four languages representing two language families (Indo-Aryan and Iranian) across three distinct orthographic systems:

1. **Urdu (`ur-PK`)**: Indo-Aryan language written in the Perso-Arabic (Nastaliq) script. Standard Pakistani Urdu vocabulary with informal social media orthography.
2. **Pashto (`ps-PK`, `ps-AF`)**: Eastern Iranian language written in an extended Arabic-Pashto alphabet containing 45 characters. Regional variations from northwestern Pakistan and Afghanistan.
3. **Roman Urdu (`ur-Latn`)**: Informal Latin-script transliteration of Urdu. Characterized by non-standardized phonetic spelling, significant code-mixing with English, and lexical borrowing.
4. **English (`en-PK`, `en-US`)**: International and Pakistani English as encountered in multilingual social networking environments.

---

## C. Provenance & Corpus Deconstruction
The dataset unifies twelve sub-corpora representing public social media interactions across Facebook, Twitter/X, and public discussion forums:

- `RU_CodeMixed_Private` (Roman Urdu, Latin)
- `RU_Private_FB_Comments` (Roman Urdu, Latin)
- `RU_Social_Media_Mix` (Roman Urdu, Latin)
- `PS_LowResource_Private` (Pashto, Arabic-Pashto)
- `PS_Private_Social_Comments` (Pashto, Arabic-Pashto)
- `PS_Public_Forum_Mix` (Pashto, Arabic-Pashto)
- `EN_Offensive_Benchmark_Private` (English, Latin)
- `EN_Public_Comments` (English, Latin)
- `EN_Social_Media_Mix` (English, Latin)
- `UR_ArabicScript_Private` (Urdu, Perso-Arabic)
- `UR_Private_Tweets` (Urdu, Perso-Arabic)
- `UR_Social_Media_Mix` (Urdu, Perso-Arabic)

*Note on Sub-Corpus Designations:* The tag `Private` indicates internal partition identifiers for datasets assembled from public web environments that were not previously distributed as independent external benchmark packages.

### Observed Intra-Group Variation Patterns
Intra-group string difference analysis across the nominal 144,265 rows identified that the nominal row expansion was characterized by four surface variation patterns:
1. **Mention Prepending**: Prepending generic account handles (e.g., `@admin`, `@friend`, `@user`, `@team`, `@newsdesk`).
2. **Dummy Hyperlink Appending**: Appending uniform placeholder URLs (e.g., `https://example.com/post`).
3. **Hashtag Permutations**: Appending a closed set of 8 generic hashtags (e.g., `#news`, `#viral`, `#opinion`, `#update`).
4. **Syntactic Slot-Filling**: Inserting domain nouns into fixed syntactic frames in Pashto and Urdu.

---

## D. Deduplication and Canonicalization Pipeline
To eliminate template leakage and cross-partition memorization, the corpus underwent a three-stage audit:
1. **Stage 1 (Exact Deduplication)**: Exact raw text matching reduced 144,265 rows to 17,559 distinct instances (87.83% reduction).
2. **Stage 2 (Normalized Clean Deduplication)**: Mention stripping, URL removal, and whitespace standardization reduced the corpus to 11,789 unique clean texts (91.83% cumulative reduction).
3. **Stage 3 (Canonical Base Consolidation)**: Collapsing intra-group variants into 1,601 canonical independence groups, retaining 1 canonical representative per group and preserving the 10,188 variants in `Dataset/audit/removed_variants.csv`.

---

## E. Audited Cross-Split Independence
Cross-split duplicate audits confirmed **zero identical pairs** between train, validation, and test partitions across five audited representations:
- Raw text
- Clean text
- Deduplication-normalized text
- Canonical base message
- Independence group ID

---

## F. Annotations and Label Harmomization
Binary classification space:
- **`offensive` (`1`)**: Abusive language, hate speech, vulgarity, personal attacks, insults, or threats. Harmonized from source labels: `offensive`, `abusive`, `hate`, `toxic`, `threatening`.
- **`non-offensive` (`0`)**: Neutral discussions, constructive criticism, benign commentary, or positive statements. Harmonized from source labels: `clean`, `neutral`, `non-offensive`.

### Class Distribution (Canonical Benchmark)
- **Offensive (`1`)**: 1,196 samples (74.70%)
- **Non-Offensive (`0`)**: 405 samples (25.30%)

---

## G. Limitations & Ethical Considerations
- **Content Warning**: The benchmark contains offensive, toxic, and insulting language solely for academic research in automated content moderation.
- **Anonymization**: All account mentions (`@user`), personal identifiers, and external hyperlinks were stripped prior to canonical release.
- **Pashto Sample Size**: The held-out test split contains 43 Pashto samples. While bootstrap estimation provides descriptive uncertainty bounds, users should note that the small sample subset limits fine-grained statistical precision on Pashto alone.
