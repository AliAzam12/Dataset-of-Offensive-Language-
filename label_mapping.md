# Label Mapping and Harmonisation Guide

This document describes how heterogeneous source labels are mapped into the unified binary benchmark labels used in this repository.

## Final benchmark labels

| Final label | Label ID | Meaning |
|---|---:|---|
| `non-offensive` | 0 | Text that does not contain offensive, abusive, toxic, hateful, insulting, threatening, or obscene content according to the original annotation |
| `offensive` | 1 | Text that contains offensive, abusive, toxic, hateful, insulting, threatening, obscene, targeted, or harmful content according to the original annotation |

## Mapping table

| Source label family | Final label | Mapping decision |
|---|---|---|
| `hate speech`, `hate`, `hateful` | `offensive` | Harmful or hateful content |
| `abusive`, `abuse` | `offensive` | Abusive content |
| `toxic`, `toxicity` | `offensive` | Toxic content |
| `offensive`, `off` | `offensive` | Direct offensive label |
| `insult`, `insulting` | `offensive` | Insulting content |
| `obscene`, `vulgar`, `profane` | `offensive` | Obscene or profane content |
| `threat`, `threatening` | `offensive` | Threatening content |
| `targeted abuse` | `offensive` | Abuse directed at a person/group |
| `neutral`, `clean`, `normal` | `non-offensive` | Non-offensive content |
| `non-toxic`, `not offensive`, `non-offensive` | `non-offensive` | Direct non-offensive label |
| dataset-specific ambiguous labels | reviewed | Mapped according to original annotation description |

## Recommended validation procedure

For transparency, ambiguous source labels should be reviewed before final mapping. The recommended process is:

1. Read the original dataset paper or documentation.
2. Identify how the original authors define each label.
3. Group semantically harmful labels into `offensive`.
4. Group benign labels into `non-offensive`.
5. Record all ambiguous labels in a mapping note.
6. If possible, ask a second reviewer to verify the mapping.

## Ambiguous label handling

Some labels may depend on context. Examples include:

| Example source label | Possible issue | Recommended action |
|---|---|---|
| `sarcasm` | May be offensive or non-offensive depending on context | Check original annotation definition |
| `intimidation` | Often harmful, but may be dataset-specific | Review original guideline |
| `provocative` | May not always be offensive | Review original guideline |
| `spam` | Not necessarily offensive | Do not map to offensive unless original dataset defines it as harmful |
| `other` | Undefined | Exclude or review manually |

## Reproducibility note

The final released files should preserve both:

- `original_label`
- `final_label`

Keeping both columns allows future researchers to inspect the original annotation scheme and reconstruct alternative label spaces if needed.
