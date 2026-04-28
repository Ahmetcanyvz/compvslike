#!/usr/bin/env bash
#SBATCH --job-name=train-correct
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_correct_%j.out
#SBATCH --error=logs/train_correct_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_train_correct.sh <TOK_NAME> <RESUME_STEP> <DATA_VARIANT>
# DATA_VARIANT = "multi" (multilingual) or "english"
# Examples:
#   sbatch run_sbatch_train_correct.sh bpe_count-multi-128k 30000 multi
#   sbatch run_sbatch_train_correct.sh greedyll-exact-128k 20000 english

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"

cd "$WORK_DIR"
pip install -e . --no-deps

TOK_NAME="${1:?missing TOK_NAME}"
RESUME_STEP="${2:?missing RESUME_STEP}"
DATA_VARIANT="${3:?missing DATA_VARIANT (multi or english)}"
SEED=42

# Source clean checkpoint from old (contaminated) outputs dir
SRC_CKPT="${WORK_DIR}/outputs/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}/.checkpoints/step${RESUME_STEP}.ckpt"
if [[ ! -f "$SRC_CKPT" ]]; then
    echo "[ERROR] Source checkpoint not found: $SRC_CKPT"
    exit 1
fi

# Data path differs for multilingual vs english
if [[ "$DATA_VARIANT" == "multi" ]]; then
    DATA_DIR="${WORK_DIR}/data/multilingual/multilingual-${TOK_NAME}"
elif [[ "$DATA_VARIANT" == "english" ]]; then
    DATA_DIR="${WORK_DIR}/data/fineweb-edu-${TOK_NAME}"
else
    echo "[ERROR] DATA_VARIANT must be 'multi' or 'english'"
    exit 1
fi

# New output dir for clean run
NEW_OUTPUT_DIR="${WORK_DIR}/outputs_correct"
RUN_DIR="${NEW_OUTPUT_DIR}/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}"
NEW_CKPT_DIR="${RUN_DIR}/.checkpoints"

# Seed the new run dir with the clean checkpoint (only if not already done)
if [[ ! -f "${NEW_CKPT_DIR}/step${RESUME_STEP}.ckpt" ]]; then
    echo "=== Copying clean checkpoint to new dir ==="
    mkdir -p "$NEW_CKPT_DIR"
    cp "$SRC_CKPT" "${NEW_CKPT_DIR}/step${RESUME_STEP}.ckpt"
fi

CONFIG_FILE="configs/correct_1B_${TOK_NAME}.yaml"

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

echo "=== Training (CORRECT): me1B-tied / ${TOK_NAME} / seed${SEED} from step${RESUME_STEP} ==="

export NCCL_DEBUG=WARN
export NCCL_NET=Socket

# Auto-resume from latest checkpoint in new dir (clean step or any newly-saved one)
LAST_CKPT=$(ls -t "$NEW_CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
echo "=== Resuming from $LAST_CKPT ==="

torchrun --nproc_per_node=4 --master_addr=localhost --master_port=29500 \
    -m src.train train "$CONFIG_FILE" --seed "$SEED" --resume "$LAST_CKPT"
