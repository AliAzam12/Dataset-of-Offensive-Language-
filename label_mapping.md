# Label Mapping and Harmonisation Guide

This document describes how heterogeneous source labels are mapped into the unified binary benchmark labels used in this repository.

## Final Benchmark Labels

| Final label | Label ID | Meaning |
|---|---:|---|
| `non-offensive` | 0 | Text that does not contain offensive, abusive, toxic, hateful, insulting, threatening, or obscene content according to the original annotation. |
| `offensive` | 1 | Text that contains offensive, abusive, toxic, hateful, insulting, threatening, obscene, targeted, or harmful content according to the original annotation. |

## Mapping Table

| Source label family | Final label | Mapping decision |
|---|---|---|
| `hate speech`, `hate`, `hateful` | `offensive` | Harmful or hateful content |
| `abusive`, `abuse` | `offensive` | Abusive content |
| `toxic`, `toxicity` | `offensive` | Toxic content |
| `offensive`, `off` | `offensive` | Direct offensive label |
| `insult`, `insulting` | `offensive` | Insulting content |
| `obscene`, `vulgar`, `profane` | `offensive` | Obscene or profane content |
| `threat`, `threatening` | `offensive` | Threatening content |
| `targeted abuse` | `offensive` | Abuse directed at a person or group |
| `neutral`, `clean`, `normal` | `non-offensive` | Non-offensive content |
| `non-toxic`, `not offensive`, `non-offensive` | `non-offensive` | Direct non-offensive label |
| Dataset-specific ambiguous labels | Reviewed | Mapped according to original annotation description |

## Ambiguous Label Handling

Some labels may depend on context and should be reviewed carefully before final mapping.

| Example source label | Possible issue | Recommended action |
|---|---|---|
| `sarcasm` | May be offensive or non-offensive depending on context | Check original annotation definition |
| `intimidation` | Often harmful, but may be dataset-specific | Review original guideline |
| `provocative` | May not always be offensive | Review original guideline |
| `spam` | Not necessarily offensive | Do not map to offensive unless original dataset defines it as harmful |
| `other` | Undefined or mixed category | Exclude or review manually |

## Recommended Validation Procedure

For transparency, ambiguous source labels should be reviewed before final mapping.

1. Read the original dataset paper or documentation.
2. Identify how the original authors define each label.
3. Group semantically harmful labels into `offensive`.
4. Group benign labels into `non-offensive`.
5. Record all ambiguous labels in a mapping note.
6. If possible, ask a second reviewer to verify the mapping.

## Notes

Label harmonisation improves comparability across datasets, but it can reduce fine-grained distinctions between hate speech, abuse, toxicity, profanity, insult, threat, and general offensive content.

Users should report that the benchmark uses a binary offensive/non-offensive label space when citing or comparing results.