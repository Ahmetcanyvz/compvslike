#!/usr/bin/env bash
#SBATCH --job-name=prep-multi
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/prep_multi_%j.out
#SBATCH --error=logs/prep_multi_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Preparing multilingual tokenized data ==="
python scripts/prepare_multilingual.py \
    -t tokenizers/bpe_count-multi-128k \
    -t tokenizers/compmax-multi-128k \
    -t tokenizers/greedyll-exact-multi-128k \
    -t tokenizers/unigramlm-multi-128k \
    -o data/multilingual \
    --total-tokens 20000000000 \
    --eng-raw-dir data/fineweb-edu-raw
echo "=== Done ==="
