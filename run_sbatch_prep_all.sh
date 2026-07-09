#!/usr/bin/env bash
#SBATCH --job-name=prep-english
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/prep_english_%j.out
#SBATCH --error=logs/prep_english_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs

echo "=== English data prep: download + tokenize 5 tokenizers @ 20B tokens ==="

python scripts/prepare_all.py \
    -t tokenizers/bpe-128k \
    -t tokenizers/compmax-128k \
    -t tokenizers/greedyll-exact-128k \
    -t tokenizers/greedyll-approx-128k \
    -t tokenizers/unigramlm-128k \
    -o data \
    --target-tokens 20000000000

echo "=== Done ==="
