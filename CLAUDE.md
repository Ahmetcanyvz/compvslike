# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

lm-trainer is a PyTorch Lightning library for training Llama-style language models with on-the-fly sequence packing. It includes an evaluation pipeline for collecting log-probabilities and regression discontinuity analysis for tokenisation bias research.

## Common Commands

```bash
# Install dependencies
uv sync                    # Basic install
uv sync --extra flash      # With flash attention
uv sync --extra dev        # With pytest

# Training
python -m src.train config.yaml
python -m src.train config.yaml --resume outputs/.checkpoints/last.ckpt

# Validation
python -m src.train validate config.yaml outputs/.checkpoints/step50000.ckpt

# Evaluation (collect log-probs)
python -m src.eval evaluate CHECKPOINT DATA_PATH OUTPUT.parquet

# Analysis (regression discontinuity)
python -m src.analysis analyze EVAL.parquet TOKENIZER_PATH CUTOFF --bandwidth 1000

# Data validation
python -m src.validate_data DATA_PATH --vocab-size 32000

# Testing
pytest                           # Run all tests
pytest tests/test_model.py       # Single file
pytest tests/test_model.py -k "test_forward"  # Single test
pytest --cov=src                 # With coverage
```

## Architecture

### Core Modules

- **`src/model.py`**: `LanguageModel` LightningModule wrapping HuggingFace `LlamaForCausalLM`. Handles:
  - Model initialization via `configure_model()` (called by Lightning)
  - Z-loss regularization in `_compute_loss()`
  - Warmup-stable-decay LR scheduler in `configure_optimizers()`
  - Weight decay exclusion for embeddings and 1D params

- **`src/data.py`**: `DataModule` and `PackedTokenDataset` for on-the-fly sequence packing:
  - `PackedTokenDataset`: Packs variable-length documents into fixed sequences with EOS tokens between docs
  - `OffsetLocator`: 2-level indexed binary search for O(log n) document lookup
  - Creates `.npy` metadata files in the data directory for shuffled indices and offsets

- **`src/train.py`**: Training CLI with custom callbacks:
  - `LogToFileCallback`: Logs to `training_log.txt` at optimizer step intervals
  - `CheckpointAtStepsCallback`: Saves at optimizer steps (not batches)

- **`src/eval.py`**: Log-probability collection with sliding window for long sequences

- **`src/analysis.py`**: Regression discontinuity analysis using weighted least squares

### Configuration

Training uses YAML config files (see `config.yaml` for template). Model architectures are in `configs/models/*.yaml`.

Key config sections:
- `paths`: tokenizer, train_data, val_data, output_dir
- `model`: config_path, use_flash_attention, torch_compile
- `training`: max_steps, batch_size, learning_rate, warmup_steps, decay_steps
- `hardware`: accelerator, devices, precision, strategy

### Data Format

Expects HuggingFace Arrow datasets with:
- `input_ids`: List[int] - variable length token sequences
- `uid` (optional): Document identifier for evaluation

## Key Implementation Details

- Flash Attention is auto-detected; falls back to SDPA
- Z-loss regularization: `logits.logsumexp(dim=-1).pow(2).mean()` scaled by `z_loss_weight`
- Learning rate schedule: linear warmup -> stable -> linear decay to `min_lr_ratio`
- Checkpoints track optimizer steps (accounting for gradient accumulation), not batches
- Evaluation uses left-padding with sliding window for sequences > window_size
