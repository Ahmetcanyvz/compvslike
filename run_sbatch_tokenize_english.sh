#!/usr/bin/env bash
#SBATCH --job-name=tok-eng
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/tok_eng_%j.out
#SBATCH --error=logs/tok_eng_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Tokenizing English (4 tokenizers, 20B tokens) ==="
python scripts/prepare_all.py \
    -t tokenizers/bpe_count-128k \
    -t tokenizers/compmax-128k \
    -t tokenizers/greedyll-exact-128k \
    -t tokenizers/unigramlm-128k \
    -o data \
    --raw-data-dir data/fineweb-edu-raw \
    --target-tokens 20000000000 \
    --num-proc 288
echo "=== Done ==="
