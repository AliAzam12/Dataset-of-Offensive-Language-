# Uncurated Nominal Raw Dataset (144,265 Records)

## Overview
This directory contains the original, un-deduplicated dataset of 144,265 nominal social media records (original_144265.csv). 

## Forensic Audit Notice
As detailed in our research paper and SOURCE_PROVENANCE_REPORT.md, this uncurated corpus exhibits severe artificial inflation (91.83% redundancy) characterized by surface template repetitions (e.g., prepended account handles @admin, @friend, dummy URLs https://example.com/post, closed-set hashtags, and slot-filling variations). 

**DO NOT USE THIS RAW FILE DIRECTLY FOR BENCHMARK EVALUATION.** Evaluating models on arbitrary random splits of this raw file results in severe cross-partition data leakage (up to 96.95% template sharing between train and test partitions) and inflates classification metrics.

For all rigorous, reproducible experiments, use the canonical benchmark provided in Dataset/canonical/ (1,601 canonical independence groups partitioned into 1,121 train, 160 validation, and 320 test samples).
