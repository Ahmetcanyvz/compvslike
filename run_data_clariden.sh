#!/usr/bin/env bash
#SBATCH --job-name=data-prep
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/data_prep_%j.out
#SBATCH --error=logs/data_prep_%j.err

set -euo pipefail

srun --container-writable --environment=lm_trainer_env bash -c '
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
'
