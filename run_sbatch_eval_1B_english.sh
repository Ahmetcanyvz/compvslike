#!/usr/bin/env bash
#SBATCH --job-name=eval-1B-eng
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3
#SBATCH --output=logs/eval_1B_eng_%A_%a.out
#SBATCH --error=logs/eval_1B_eng_%A_%a.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

WORK=/iopsstor/scratch/cscs/ayavuz/compvslike
TOKS=(bpe_count-128k compmax-128k greedyll-exact-128k unigramlm-128k)
TOK=${TOKS[$SLURM_ARRAY_TASK_ID]}

MODEL="me1B-tied_${TOK}_20Btok_seed42"
CKPT="${WORK}/outputs/${MODEL}/.checkpoints/last.ckpt"
TOKENIZER="${WORK}/tokenizers/${TOK}"
RAW_TEST="${WORK}/data/fineweb-edu-raw/test"
OUT_DIR="${WORK}/eval_results_1B/${MODEL}"

mkdir -p "$OUT_DIR"
echo "=== BPB eval: ${MODEL} (tokenizer ${TOK}) ==="
echo "  ckpt: ${CKPT}"
[[ -f "$CKPT" ]] || { echo "!! checkpoint missing"; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "!! tokenizer missing: $TOKENIZER"; exit 1; }

python -m src.eval_bpb "$CKPT" "$TOKENIZER" "$RAW_TEST" -o "${OUT_DIR}/bpb.parquet"
echo "=== Done: ${MODEL} ==="
