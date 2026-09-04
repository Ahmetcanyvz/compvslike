#!/usr/bin/env bash
# Host-side SLURM array job: seeds x tokenizers, one node (4 GPUs, DDP) per task.
# Mirrors run_sbatch_train_1B_seeds.sh.
#
# Submit flags come from env.local.sh; see slurm/submit.sh.
#
#SBATCH --job-name=cvl-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=logs_training/train_%A_%a.out
#SBATCH --error=logs_training/train_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#
# NOTE: deliberately no --environment here. The batch script and the requeue
# trap run on the HOST, where scontrol works. The container's glibc is too old
# for the host SLURM libraries, so scontrol must NOT run inside the container.

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/env.sh"
mkdir -p "${CVL_ROOT}/logs_training" "${CVL_ROOT}/configs/generated"

# Auto-requeue ~180s before walltime. Runs on the host, so scontrol works.
requeue_handler() {
    echo "=== [$(date)] USR1 near walltime — requeuing ${SLURM_JOB_ID} ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    kill "${SRUN_PID:-}" 2>/dev/null || true
}
trap requeue_handler USR1

MODEL="${MODEL:-me1B-tied}"
BUDGET_B="${BUDGET_B:-20}"
read -r -a SEEDS <<< "${SEEDS:-43 44}"
read -r -a TOKS  <<< "${TOKS:-bpe_count-128k compmax-128k greedyll-exact-128k unigramlm-128k}"

n_tok=${#TOKS[@]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / n_tok))]}
TOK_NAME=${TOKS[$((SLURM_ARRAY_TASK_ID % n_tok))]}

CONFIG_FILE="${CVL_ROOT}/configs/generated/slurm_${MODEL}_${TOK_NAME}_seed${SEED}.yaml"
cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${CVL_TOKENIZERS}/${TOK_NAME}
  train_data: ${CVL_DATA}/fineweb-edu-${TOK_NAME}/train
  val_data: ${CVL_DATA}/fineweb-edu-${TOK_NAME}/val
  test_data: ${CVL_DATA}/fineweb-edu-${TOK_NAME}/test
  output_dir: ${CVL_OUTPUTS}

model:
  config_path: ${CVL_ROOT}/configs/models/${MODEL}.yaml
  use_flash_attention: true
  use_liger_kernel: true
  torch_compile: false

training:
  seed: ${SEED}
  max_tokens: ${BUDGET_B}000000000
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

# Resume flag computed on the host (shared filesystem).
CKPT_DIR="${CVL_OUTPUTS}/${MODEL}_${TOK_NAME}_${BUDGET_B}Btok_seed${SEED}/.checkpoints"
RESUME_FLAG=""
if [[ -d "$CKPT_DIR" ]]; then
    LAST_CKPT=$(ls -t "$CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
    [[ -n "$LAST_CKPT" ]] && { RESUME_FLAG="--resume $LAST_CKPT"; echo "=== Resuming from $LAST_CKPT ==="; }
fi

PORT=$((29500 + SLURM_ARRAY_TASK_ID))

# Workload runs in the container; the batch shell stays on the host
# (background + wait) so the USR1 trap can fire and requeue mid-training.
srun ${CVL_CONTAINER_ENV:+--environment="$CVL_CONTAINER_ENV"} --container-writable \
    --nodes=1 --ntasks=1 \
    bash "${CVL_ROOT}/slurm/train_inner.sh" "$SEED" "$TOK_NAME" "$CONFIG_FILE" "$PORT" $RESUME_FLAG &
SRUN_PID=$!
wait "$SRUN_PID"
TRAIN_RC=$?

echo "=== Exit ${TRAIN_RC}: ${MODEL} / ${TOK_NAME} / seed${SEED} ==="
exit $TRAIN_RC
