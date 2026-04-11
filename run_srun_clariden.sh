#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
DATA_BASE="${WORK_DIR}/data"

cd "$WORK_DIR"
pip install -e . --no-deps

TOK_TYPE="${1:-greedyll-exact}"
VOCAB="128k"
SEED=42
TOK_NAME="${TOK_TYPE}-${VOCAB}"

CONFIG_FILE="configs/clariden_1B_${TOK_NAME}.yaml"

cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${TOK_NAME}
  train_data: ${DATA_BASE}/fineweb-edu-${TOK_NAME}/train
  val_data: ${DATA_BASE}/fineweb-edu-${TOK_NAME}/val
  test_data: ${DATA_BASE}/fineweb-edu-${TOK_NAME}/test
  output_dir: ${WORK_DIR}/outputs

model:
  config_path: models/me1B-tied.yaml
  use_flash_attention: true
  use_liger_kernel: true
  torch_compile: false

training:
  seed: ${SEED}
  max_tokens: 20000000000
  batch_size: 16
  eval_batch_size: 4
  gradient_accumulation: 2
  sequence_length: 2048
  learning_rate: 6.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  warmup_steps: 2000
  decay_steps: 2000
  min_lr_ratio: 0.01
  z_loss_weight: 0

checkpoint:
  save_every_n_steps: 5000
  save_dir: .checkpoints
  save_last: true

hardware:
  accelerator: gpu
  devices: 4
  precision: bf16-true
  strategy: ddp

logging:
  log_every_n_steps: 50
  log_loss_every_n_steps: 1000
  val_check_interval: 1.0
EOF

echo "=== Pre-creating dataset metadata ==="
python -c "
from src.data import PackedTokenDataset
for split in ['train', 'val']:
    path = '${DATA_BASE}/fineweb-edu-${TOK_NAME}/' + split
    print(f'Creating metadata for {path}...')
    ds = PackedTokenDataset(path, seq_len=2048, eos_token_id=0, shuffle_seed=42 if split == 'train' else None)
    print(f'  {len(ds)} sequences')
"

echo "=== Training: me1B-tied / ${TOK_NAME} / seed${SEED} ==="

export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_TIMEOUT=3600
export MASTER_ADDR=localhost
export MASTER_PORT=29500

python -m src.train train "$CONFIG_FILE" --seed "$SEED"
