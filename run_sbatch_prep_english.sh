#!/usr/bin/env bash
#SBATCH --job-name=prep-eng
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/prep_eng_%j.out
#SBATCH --error=logs/prep_eng_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

# Original pipeline (scripts/prepare_all.py):
#   download_base_data      -> base 2B, seed-42 shuffle, 95/2.5/2.5 -> val/test = 47,384 each
#   download_extra_train_data -> +18B into train_extra/ (chunked, resumable)
#   then tokenize with all 4 tokenizers
# No --raw-data-dir (fresh download), no --tokenize-only. Resumable across 12h jobs:
#   re-submitting skips the finished base and resumes extra chunks.
echo "=== Original prepare_all: download base 2B + extra 18B + tokenize ==="
python scripts/prepare_all.py \
    -t tokenizers/bpe_count-128k \
    -t tokenizers/compmax-128k \
    -t tokenizers/greedyll-exact-128k \
    -t tokenizers/unigramlm-128k \
    -o data \
    --target-tokens 20000000000 \
    --num-proc 32
echo "=== Done ==="
