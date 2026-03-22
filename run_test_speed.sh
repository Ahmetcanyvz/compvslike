#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
DATA_BASE="/local/home/ayavuz/compvslike/data"
TOK="bpe-128k"

TORCH_LIB=$(uv run --no-sync python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null) || true
[ -n "${TORCH_LIB:-}" ] && export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

CONFIG_FILE="configs/speed_test.yaml"

cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${TOK}
  train_data: ${DATA_BASE}/fineweb-edu-${TOK}/train
  val_data: ${DATA_BASE}/fineweb-edu-${TOK}/val
  test_data: ${DATA_BASE}/fineweb-edu-${TOK}/test
  output_dir: ./outputs_speed_test

model:
  config_path: models/me1B-tied.yaml
  use_flash_attention: true
  use_liger_kernel: true
  torch_compile: true

training:
  seed: 42
  max_steps: 100
  batch_size: 8
  gradient_accumulation: 16
  sequence_length: 2048
  learning_rate: 6.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  warmup_steps: 10
  decay_steps: 10
  min_lr_ratio: 0.01
  z_loss_weight: 0

checkpoint:
  save_every_n_steps: 10000
  save_dir: .checkpoints
  save_last: false

hardware:
  accelerator: gpu
  devices: 1
  precision: bf16-true
  strategy: auto

logging:
  log_every_n_steps: 10
  log_loss_every_n_steps: 50
EOF

uv run --no-sync python -m src.train train "$CONFIG_FILE" --seed 42
