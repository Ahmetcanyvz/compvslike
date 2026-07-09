#!/usr/bin/env bash
#SBATCH --job-name=eval-1B-greedyll
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/eval_greedyll_%j.out
#SBATCH --error=logs/eval_greedyll_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps

CHECKPOINT="outputs/me1B-tied_greedyll-exact-128k_20Btok_seed42/.checkpoints/last.ckpt"
TOKENIZER="tokenizers/greedyll-exact-128k"
RAW_TEST="data/fineweb-edu-raw/test"
OUT="eval_results/me1B-tied_greedyll-exact-128k_20Btok_seed42"

mkdir -p "$OUT"

echo "=== BLiMP ==="
python -m src.eval_blimp "$CHECKPOINT" "$TOKENIZER" -o "$OUT/blimp.parquet"

echo "=== BPB ==="
python -m src.eval_bpb "$CHECKPOINT" "$TOKENIZER" "$RAW_TEST" -o "$OUT/bpb.parquet"

echo "=== Done ==="
