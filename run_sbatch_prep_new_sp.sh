#!/usr/bin/env bash
#SBATCH --job-name=prep-new-sp
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/prep_new_sp_%j.out
#SBATCH --error=logs/prep_new_sp_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Tokenize English data with the two new SentencePiece-based tokenizers.
# Uses existing fineweb-edu-raw (no re-download needed).

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs

echo "=== Tokenizing English data with new SentencePiece tokenizers (20B target) ==="

python scripts/prepare_all.py \
    -t tokenizers/compmax_sentencepiece-128k \
    -t tokenizers/unigramlm_sentencepiece-128k \
    -o data \
    --raw-data-dir data/fineweb-edu-raw \
    --target-tokens 20000000000

echo "=== Done ==="
