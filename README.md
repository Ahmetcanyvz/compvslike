# lm-trainer

A clean, self-contained library for training language models with PyTorch Lightning.

## Features

- **Simple YAML configuration** - No complex config frameworks
- **On-the-fly sequence packing** - Efficient variable-length document handling
- **Multiple model sizes** - 57M, 100M, 340M, 850M parameter configs
- **Multi-GPU support** - DDP, FSDP via Lightning
- **Flash Attention** - Automatic detection and usage
- **Evaluation pipeline** - Log-probability collection with sliding window
- **Analysis tools** - Regression discontinuity analysis for tokenisation bias

## Installation

```bash
# Using uv (recommended)
uv sync

# With flash attention
uv sync --extra flash

# With dev dependencies (pytest)
uv sync --extra dev
```

## Quick Start

### 1. Validate your data

```bash
python -m src.validate_data path/to/tokenized_data --vocab-size 32000
```

### 2. Configure training

Edit `config.yaml`:

```yaml
paths:
  tokenizer: path/to/tokenizer
  train_data: path/to/train
  val_data: path/to/val
  output_dir: ./outputs

model:
  config_path: configs/models/me57M-tied.yaml

training:
  max_steps: 50000
  batch_size: 32
  learning_rate: 3.0e-4
```

### 3. Train

```bash
python -m src.train config.yaml
```

Resume from checkpoint:

```bash
python -m src.train config.yaml --resume outputs/.checkpoints/last.ckpt
```

### 4. Evaluate

Collect log-probabilities:

```bash
python -m src.eval evaluate \
    outputs/.checkpoints/step50000.ckpt \
    path/to/test_data \
    outputs/eval_results.parquet
```

### 5. Analyze

Run regression discontinuity analysis:

```bash
python -m src.analysis analyze \
    outputs/eval_results.parquet \
    path/to/tokenizer \
    32000 \
    --bandwidth 1000
```

## Model Configurations

| Config | Parameters | Hidden | Layers | Heads | Tied |
|--------|-----------|--------|--------|-------|------|
| me57M-tied | 57M | 768 | 6 | 24 | Yes |
| me100M-tied | 100M | 576 | 30 | 9 | Yes |
| me100M | 100M | 576 | 30 | 9 | No |
| me340M-tied | 340M | 1024 | 32 | 15 | Yes |
| me850M | 850M | 1536 | 24 | 32 | No |

## Data Format

The dataset should be a HuggingFace Arrow dataset with:
- `input_ids`: List of token IDs (variable length)
- `uid` (optional): Unique document identifier

Example structure:
```
data/
├── train/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
└── val/
    └── ...
```

## Configuration Reference

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 50000 | Total training steps |
| `batch_size` | 32 | Per-device batch size |
| `learning_rate` | 3e-4 | Peak learning rate |
| `warmup_steps` | 2000 | LR warmup steps |
| `decay_steps` | 10000 | LR decay steps |
| `weight_decay` | 0.1 | AdamW weight decay |
| `z_loss_weight` | 1e-4 | Z-loss regularization |

### Hardware

| Parameter | Default | Description |
|-----------|---------|-------------|
| `accelerator` | gpu | Device type |
| `devices` | auto | Number of devices |
| `precision` | bf16-mixed | Training precision |
| `strategy` | auto | Distributed strategy |

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src
```

## Project Structure

```
lm-trainer/
├── config.yaml           # Example config
├── configs/models/       # Model architecture configs
├── src/
│   ├── data.py          # DataModule, PackedTokenDataset
│   ├── model.py         # LanguageModel LightningModule
│   ├── train.py         # Training CLI
│   ├── eval.py          # Evaluation CLI
│   ├── analysis.py      # RD analysis CLI
│   └── validate_data.py # Data validation CLI
└── tests/               # Test suite
```

## CLI Commands

### Training

```bash
python -m src.train CONFIG_PATH [--resume CHECKPOINT]
python -m src.train validate CONFIG_PATH CHECKPOINT
```

### Evaluation

```bash
python -m src.eval evaluate CHECKPOINT DATA_PATH OUTPUT
python -m src.eval aggregate INPUT_DIR OUTPUT
```

### Analysis

```bash
python -m src.analysis analyze EVAL_PATH TOKENIZER_PATH CUTOFF
python -m src.analysis token-stats EVAL_PATH OUTPUT
python -m src.analysis compare-models EVAL_PATH1 EVAL_PATH2 --tokenizer PATH --cutoff N
```

### Data Validation

```bash
python -m src.validate_data validate DATA_PATH [--vocab-size N]
python -m src.validate_data compare TRAIN_PATH VAL_PATH [--test TEST_PATH]
```

## License

MIT
