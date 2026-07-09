#!/usr/bin/env bash
#SBATCH --job-name=train-scratch
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_scratch_%j.out
#SBATCH --error=logs/train_scratch_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_train_scratch.sh <TOK_NAME>
# Trains a 1B model on English data with the given tokenizer, from scratch.
# Auto-resumes from the latest checkpoint in outputs_correct/ if one exists.
# Example:
#   sbatch run_sbatch_train_scratch.sh compmax_sentencepiece-128k
#   sbatch run_sbatch_train_scratch.sh unigramlm_sentencepiece-128k

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"

cd "$WORK_DIR"
pip install -e . --no-deps

TOK_NAME="${1:?missing TOK_NAME}"
SEED=42

DATA_DIR="${WORK_DIR}/data/fineweb-edu-${TOK_NAME}"
if [[ ! -d "${DATA_DIR}/train" ]]; then
    echo "[ERROR] Tokenized data not found at ${DATA_DIR}/train"
    echo "        Run data prep first: sbatch run_sbatch_prep_new_sp.sh"
    exit 1
fi

NEW_OUTPUT_DIR="${WORK_DIR}/outputs_correct"
RUN_DIR="${NEW_OUTPUT_DIR}/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}"
NEW_CKPT_DIR="${RUN_DIR}/.checkpoints"
mkdir -p "$NEW_CKPT_DIR"

CONFIG_FILE="configs/scratch_1B_${TOK_NAME}.yaml"

cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${TOK_NAME}
  train_data: ${DATA_DIR}/train
  val_data: ${DATA_DIR}/val
  test_data: ${DATA_DIR}/test
  output_dir: ${NEW_OUTPUT_DIR}

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
  val_check_interval: 20000
EOF

echo "=== Pre-creating dataset metadata ==="
python -c "
from src.data import PackedTokenDataset
for split in ['train', 'val']:
    path = '${DATA_DIR}/' + split
    print(f'Creating metadata for {path}...')
    ds = PackedTokenDataset(path, seq_len=2048, eos_token_id=0, shuffle_seed=42 if split == 'train' else None)
    print(f'  {len(ds)} sequences')
"

export NCCL_DEBUG=WARN
export NCCL_NET=Socket

# Auto-resume from latest checkpoint if any
LAST_CKPT=$(ls -t "$NEW_CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
RESUME_FLAG=""
if [[ -n "$LAST_CKPT" && -f "$LAST_CKPT" ]]; then
    echo "=== Resuming from $LAST_CKPT ==="
    RESUME_FLAG="--resume $LAST_CKPT"
else
    echo "=== Training from scratch: me1B-tied / ${TOK_NAME} / seed${SEED} ==="
fi

torchrun --nproc_per_node=4 --master_addr=localhost --master_port=29500 \
    -m src.train train "$CONFIG_FILE" --seed "$SEED" $RESUME_FLAG
