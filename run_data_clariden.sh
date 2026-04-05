#!/usr/bin/env bash
#SBATCH --job-name=data-prep
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

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
