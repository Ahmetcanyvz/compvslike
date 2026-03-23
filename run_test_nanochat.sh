#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
DATA_BASE="/local/home/ayavuz/compvslike/data"
TOK="bpe-128k"

TORCH_LIB=$(uv run --no-sync python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null) || true
[ -n "${TORCH_LIB:-}" ] && export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

CONFIG_FILE="configs/test_nanochat.yaml"

cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${TOK}
  train_data: ${DATA_BASE}/fineweb-edu-${TOK}/train
  val_data: ${DATA_BASE}/fineweb-edu-${TOK}/val
  test_data: ${DATA_BASE}/fineweb-edu-${TOK}/test
  output_dir: ./outputs_test_nanochat

model:
  nanochat_model: nc1B
  window_pattern: SSSL

training:
  seed: 42
  max_steps: 100
  batch_size: 8
  gradient_accumulation: 16
  sequence_length: 2048

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

uv run --no-sync python -m src.train_nanochat "$CONFIG_FILE" --seed 42
