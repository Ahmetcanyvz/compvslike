#!/usr/bin/env bash
#SBATCH --job-name=eval-multiblimp
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/eval_multiblimp_%j.out
#SBATCH --error=logs/eval_multiblimp_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_eval_multiblimp.sh <TOK_NAME>
# Evaluates one multilingual 1B model on MultiBLiMP for eng/deu/spa/tur.
# (MultiBLiMP has no Chinese/cmn config, so Chinese is not covered.)
# Example:
#   sbatch run_sbatch_eval_multiblimp.sh bpe_count-multi-128k

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
OUTPUTS="${WORK_DIR}/outputs_correct"
EVAL_OUT="${WORK_DIR}/eval_results_correct"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs "$EVAL_OUT"

TOK_NAME="${1:?missing TOK_NAME (e.g. bpe_count-multi-128k)}"
SEED=42

MODEL="me1B-tied_${TOK_NAME}_20Btok_seed${SEED}"
checkpoint="${OUTPUTS}/${MODEL}/.checkpoints/last.ckpt"
tokenizer="${TOKENIZER_BASE}/${TOK_NAME}"
out_dir="${EVAL_OUT}/${MODEL}"

if [[ ! -f "$checkpoint" ]]; then
    echo "[ERROR] Checkpoint not found: $checkpoint"
    exit 1
fi

mkdir -p "$out_dir"
echo "=== MultiBLiMP eval: $MODEL ==="

LANGS=(eng deu spa tur)
for lang in "${LANGS[@]}"; do
    out_file="$out_dir/multiblimp_${lang}.parquet"
    if [[ -f "$out_file" ]]; then
        echo "  [SKIP] $lang done"
    else
        echo "  [RUN] $lang..."
        python -m src.eval_multiblimp "$checkpoint" "$tokenizer" "$lang" -o "$out_file"
    fi
done

echo "=== Done ==="
