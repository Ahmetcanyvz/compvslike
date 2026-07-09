#!/usr/bin/env bash
#SBATCH --job-name=eval-1B-uni
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/eval_1B_uni_%j.out
#SBATCH --error=logs/eval_1B_uni_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
RAW_TEST="${WORK_DIR}/data/fineweb-edu-raw/test"
OUTPUTS="${WORK_DIR}/outputs"
EVAL_OUT="${WORK_DIR}/eval_results"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs "$EVAL_OUT"

MODEL="me1B-tied_unigramlm-128k_20Btok_seed42"
TOK_NAME="unigramlm-128k"

checkpoint="${OUTPUTS}/${MODEL}/.checkpoints/last.ckpt"
tokenizer="${TOKENIZER_BASE}/${TOK_NAME}"
out_dir="${EVAL_OUT}/${MODEL}"

if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] Checkpoint not found: $checkpoint"
    exit 1
fi

mkdir -p "$out_dir"
echo "=== Evaluating: $MODEL ==="

if [[ -f "$out_dir/blimp.parquet" ]]; then
    echo "  [SKIP] BLiMP done"
else
    echo "  [RUN] BLiMP..."
    python -m src.eval_blimp "$checkpoint" "$tokenizer" -o "$out_dir/blimp.parquet"
fi

if [[ -f "$out_dir/bpb.parquet" ]]; then
    echo "  [SKIP] BPB done"
else
    echo "  [RUN] BPB..."
    python -m src.eval_bpb "$checkpoint" "$tokenizer" "$RAW_TEST" -o "$out_dir/bpb.parquet"
fi

echo "=== Done ==="
