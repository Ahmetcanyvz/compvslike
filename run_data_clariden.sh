#!/usr/bin/env bash
#SBATCH --job-name=data-prep
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err
#SBATCH --environment=lm_trainer_env
#SBATCH --container-writable

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps

mkdir -p logs

echo "=== Starting data preparation ==="
python scripts/prepare_all.py \
    -t tokenizers/greedyll-exact-128k \
    -t tokenizers/unigramlm-128k \
    -o data \
    --target-tokens 20000000000

echo "=== Data preparation complete ==="
