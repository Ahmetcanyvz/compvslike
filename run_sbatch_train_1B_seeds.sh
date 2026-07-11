#!/usr/bin/env bash
#SBATCH --job-name=train-1B-s4344
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --array=0-7
#SBATCH --output=logs_training/train_1B_s4344_%A_%a.out
#SBATCH --error=logs_training/train_1B_s4344_%A_%a.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -uo pipefail

# Auto-requeue ~180s before the walltime: SLURM sends USR1 to this batch shell,
# we requeue the job, and on restart it auto-resumes from the last checkpoint.
requeue_handler() {
    echo "=== [$(date)] USR1 near walltime — requeuing ${SLURM_JOB_ID} ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    kill "${TRAIN_PID:-}" 2>/dev/null || true
}
trap requeue_handler USR1

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
DATA_BASE="${WORK_DIR}/data"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs_training

# 8 jobs = seeds {43,44} x 4 tokenizers. One node (4 GPUs, DDP) per job.
SEEDS=(43 44)
TOKS=(bpe_count-128k compmax-128k greedyll-exact-128k unigramlm-128k)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 4))]}
TOK_NAME=${TOKS[$((SLURM_ARRAY_TASK_ID % 4))]}

CONFIG_FILE="configs/clariden_1B_${TOK_NAME}_seed${SEED}.yaml"

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
  save_every_n_steps: 2000
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

# Pre-create dataset metadata single-process (shuffle_seed = training seed),
# so the 4 DDP ranks load it instead of racing to create it.
echo "=== [seed ${SEED} / ${TOK_NAME}] Pre-creating dataset metadata ==="
python -c "
from src.data import PackedTokenDataset
for split, sd in [('train', ${SEED}), ('val', None)]:
    path = '${DATA_BASE}/fineweb-edu-${TOK_NAME}/' + split
    ds = PackedTokenDataset(path, seq_len=2048, eos_token_id=0, shuffle_seed=sd)
    print(f'  {split}: {len(ds)} sequences')
"

echo "=== Training: me1B-tied / ${TOK_NAME} / seed${SEED} ==="

export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_TIMEOUT=3600
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

# Auto-resume from last checkpoint if it exists (resubmit to continue past 12h).
CKPT_DIR="${WORK_DIR}/outputs/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}/.checkpoints"
RESUME_FLAG=""
if [[ -d "$CKPT_DIR" ]]; then
    LAST_CKPT=$(ls -t "$CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
    if [[ -n "$LAST_CKPT" ]]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "=== Resuming from $LAST_CKPT ==="
    fi
fi

# Run in background + wait so the USR1 trap can fire mid-training.
torchrun --nproc_per_node=4 --master_addr=localhost --master_port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    -m src.train train "$CONFIG_FILE" --seed "$SEED" $RESUME_FLAG &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_RC=$?

echo "=== Exit ${TRAIN_RC}: me1B-tied / ${TOK_NAME} / seed${SEED} ==="
exit $TRAIN_RC
