#!/usr/bin/env bash
#SBATCH --job-name=lm-train-1B
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

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
  batch_size: 32
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
  val_check_interval: 0.5
EOF

echo "=== Training: me1B-tied / ${TOK_NAME} / seed${SEED} ==="
echo "=== GPUs: 4 x GH200 ==="
echo "=== Config: ${CONFIG_FILE} ==="

mkdir -p logs

python -m src.train train "$CONFIG_FILE" --seed "$SEED"

echo "=== Training complete ==="
