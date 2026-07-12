#!/usr/bin/env bash
#SBATCH --job-name=traineval-1B-s4344
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --array=0-7
#SBATCH --output=logs_training/traineval_1B_s4344_%A_%a.out
#SBATCH --error=logs_training/traineval_1B_s4344_%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
# Batch script + trap run on the HOST (scontrol works). Training & eval run in
# the container via srun --environment. Resumes training; when a model actually
# reaches max_tokens (not a requeue), it runs test-set BPB right after.

set -uo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
DATA_BASE="${WORK_DIR}/data"
mkdir -p "${WORK_DIR}/logs_training"

# Auto-requeue ~180s before walltime; REQUEUED tells us NOT to eval this cycle.
REQUEUED=0
requeue_handler() {
    REQUEUED=1
    echo "=== [$(date)] USR1 near walltime — requeuing ${SLURM_JOB_ID} ==="
    scontrol requeue "${SLURM_JOB_ID}" || true
    kill "${SRUN_PID:-}" 2>/dev/null || true
}
trap requeue_handler USR1

# 8 jobs = seeds {43,44} x 4 tokenizers. One node (4 GPUs, DDP) per job.
SEEDS=(43 44)
TOKS=(bpe_count-128k compmax-128k greedyll-exact-128k unigramlm-128k)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 4))]}
TOK_NAME=${TOKS[$((SLURM_ARRAY_TASK_ID % 4))]}

CONFIG_FILE="${WORK_DIR}/configs/clariden_1B_${TOK_NAME}_seed${SEED}.yaml"
mkdir -p "${WORK_DIR}/configs"
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

CKPT_DIR="${WORK_DIR}/outputs/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}/.checkpoints"
RESUME_FLAG=""
if [[ -d "$CKPT_DIR" ]]; then
    LAST_CKPT=$(ls -t "$CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
    if [[ -n "$LAST_CKPT" ]]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "=== Resuming from $LAST_CKPT ==="
    fi
fi

PORT=$((29500 + SLURM_ARRAY_TASK_ID))

# ---- Train (in container; host waits so the USR1 trap can fire) ----
srun --environment=lm_trainer_env --container-writable --nodes=1 --ntasks=1 \
    bash "${WORK_DIR}/train_1B_inner.sh" "$SEED" "$TOK_NAME" "$CONFIG_FILE" "$PORT" $RESUME_FLAG &
SRUN_PID=$!
wait "$SRUN_PID"
TRAIN_RC=$?

if [[ "$REQUEUED" -eq 1 ]]; then
    echo "=== Requeued; training not finished. Skipping eval this cycle. ==="
    exit 0
fi
if [[ "$TRAIN_RC" -ne 0 ]]; then
    echo "=== Training exited ${TRAIN_RC} without requeue; NOT evaluating. ==="
    exit "$TRAIN_RC"
fi

# ---- Training finished -> test-set BPB (only runs on completion) ----
EVAL_OUT="${WORK_DIR}/eval_results_1B/me1B-tied_${TOK_NAME}_20Btok_seed${SEED}"
mkdir -p "$EVAL_OUT"
if [[ -f "${EVAL_OUT}/bpb.parquet" ]]; then
    echo "=== BPB already exists at ${EVAL_OUT}/bpb.parquet; skipping eval. ==="
    exit 0
fi

FINAL_CKPT=$(ls -t "$CKPT_DIR"/last.ckpt "$CKPT_DIR"/step*.ckpt 2>/dev/null | head -1)
echo "=== Training complete. Test BPB for ${TOK_NAME} seed${SEED} using ${FINAL_CKPT} ==="
srun --environment=lm_trainer_env --container-writable --nodes=1 --ntasks=1 bash -c "
set -e
cd '${WORK_DIR}'
pip install -e . --no-deps
python -m src.eval_bpb '${FINAL_CKPT}' '${TOKENIZER_BASE}/${TOK_NAME}' '${DATA_BASE}/fineweb-edu-raw/test' -o '${EVAL_OUT}/bpb.parquet'
"
echo "=== Done: trained + BPB for me1B-tied / ${TOK_NAME} / seed${SEED} -> ${EVAL_OUT}/bpb.parquet ==="
