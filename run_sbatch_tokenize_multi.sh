#!/usr/bin/env bash
#SBATCH --job-name=tok-multi
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/tok_multi_%j.out
#SBATCH --error=logs/tok_multi_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Tokenizing multilingual (4 tokenizers, 20B tokens) ==="
python scripts/prepare_multilingual.py \
    -t tokenizers/bpe_count-multi-128k \
    -t tokenizers/compmax-multi-128k \
    -t tokenizers/greedyll-exact-multi-128k \
    -t tokenizers/unigramlm-multi-128k \
    -o data/multilingual \
    --raw-data-dir data/multilingual-raw \
    --total-tokens 20000000000 \
    --eng-raw-dir data/fineweb-edu-raw
echo "=== Done ==="
