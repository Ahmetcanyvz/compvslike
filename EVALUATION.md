# Evaluation Guide

This document explains the evaluation metrics available for comparing language models trained with different tokenizers.

## Overview

When comparing models trained with different vocabulary sizes (e.g., 8k, 32k, 128k), standard perplexity is not directly comparable because:
- Different tokenizers produce different numbers of tokens for the same text
- A model with smaller vocabulary has more tokens to predict (artificially higher perplexity)

We provide two evaluation metrics that enable fair comparison:

1. **Bits-per-byte (BPB)** - Normalizes by text length in bytes
2. **BLiMP** - Tests grammatical knowledge via minimal pairs

---

## Bits-per-Byte (BPB)

### What it measures

BPB measures how many bits the model needs, on average, to encode each byte of text. Lower is better.

### Why use it

- **Fair comparison across tokenizers**: Normalizes by byte length, not token count
- **Interpretable**: A BPB of 1.0 means 1 bit per byte; typical values are 0.8-1.5
- **Compression perspective**: Relates to how well the model can compress text

### Formula

```
BPB = total_cross_entropy_loss / (num_bytes × ln(2))
```

Where:
- `total_cross_entropy_loss` is in nats (natural log units)
- `num_bytes` is the UTF-8 byte length of the text
- `ln(2) ≈ 0.693` converts nats to bits

### Usage

```bash
# Basic usage
uv run python -m src.eval_bpb \
    checkpoint.ckpt \
    /path/to/tokenizer \
    /path/to/raw_text_data

# With options
uv run python -m src.eval_bpb \
    checkpoint.ckpt \
    /path/to/tokenizer \
    /path/to/raw_text_data \
    --output results.parquet \
    --max-samples 1000 \
    --device cuda
```

### Requirements

- Dataset must have a `text` column with raw text (not tokenized)
- Use the same test set for all models being compared

### Example output

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric            ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Total documents   │ 10,000      │
│ Total bytes       │ 45,678,901  │
│ Total tokens      │ 12,345,678  │
│ Bytes per token   │ 3.70        │
│                   │             │
│ Aggregate BPB     │ 1.0842      │
│ Mean doc BPB      │ 1.0915      │
│ Std doc BPB       │ 0.1523      │
│ Min doc BPB       │ 0.6234      │
│ Max doc BPB       │ 2.3456      │
└───────────────────┴─────────────┘
```

---

## BLiMP (Benchmark of Linguistic Minimal Pairs)

### What it measures

BLiMP tests whether the model has learned grammatical rules by presenting minimal pairs of sentences:
- One grammatically correct
- One with a grammatical error

The model should assign higher probability to the grammatical sentence.

### Why use it

- **Grammar understanding**: Tests 67 specific linguistic phenomena
- **Tokenizer-agnostic**: Compares sentence probabilities, not token-level metrics
- **Interpretable**: Accuracy percentage for each phenomenon

### Linguistic phenomena tested

BLiMP covers 12 categories:

1. **Anaphor agreement** - Reflexive pronoun agreement
2. **Argument structure** - Verb argument requirements
3. **Binding** - Principle A (anaphors), Principle B (pronouns)
4. **Control/raising** - Tough-movement, raising verbs
5. **Determiner-noun agreement** - Article-noun agreement
6. **Ellipsis** - N-bar ellipsis
7. **Filler-gap** - Wh-movement dependencies
8. **Irregular forms** - Irregular plurals, past participles
9. **Island effects** - Extraction constraints
10. **NPI licensing** - Negative polarity items
11. **Quantifiers** - Superlatives, existentials
12. **Subject-verb agreement** - Number agreement

### Usage

```bash
# Evaluate all 67 tasks
uv run python -m src.eval_blimp \
    checkpoint.ckpt \
    /path/to/tokenizer

# Evaluate specific tasks
uv run python -m src.eval_blimp \
    checkpoint.ckpt \
    /path/to/tokenizer \
    --tasks "anaphor_number_agreement,subject_verb_agreement_1"

# Save results
uv run python -m src.eval_blimp \
    checkpoint.ckpt \
    /path/to/tokenizer \
    --output blimp_results.parquet
```

### Example output

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Task                                    ┃ Accuracy ┃ Correct ┃ Total ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ anaphor_number_agreement                │    89.2% │     892 │  1000 │
│ determiner_noun_agreement_1             │    85.4% │     854 │  1000 │
│ regular_plural_subject_verb_agreement_1 │    78.3% │     783 │  1000 │
│ ...                                     │      ... │     ... │   ... │
│                                         │          │         │       │
│ OVERALL                                 │    67.5% │   45123 │ 66900 │
└─────────────────────────────────────────┴──────────┴─────────┴───────┘
```

---

## Comparing Models with Different Tokenizers

### Recommended workflow

1. Train models with same architecture but different tokenizers (8k, 32k, 128k vocab)
2. Evaluate each on the **same** test set
3. Compare BPB (lower = better compression)
4. Compare BLiMP accuracy (higher = better grammar)

### Example comparison

| Tokenizer | BPB   | BLiMP Accuracy |
|-----------|-------|----------------|
| BPE-8k    | 1.15  | 62.3%          |
| BPE-32k   | 1.08  | 67.5%          |
| BPE-128k  | 1.12  | 65.1%          |

### Interpretation

- **BPB differences**: Smaller differences (< 0.05) may not be significant
- **BLiMP differences**: Look for consistent patterns across task categories
- **Trade-offs**: Larger vocab may help compression but require more parameters

---

## References

- Warstadt, A., et al. (2020). "BLiMP: The Benchmark of Linguistic Minimal Pairs for English"
- Gao, L., et al. (2020). "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" (BPB metrics)
