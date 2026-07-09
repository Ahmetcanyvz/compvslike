#!/usr/bin/env bash
#SBATCH --job-name=dl-multilingual
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/dl_multi_%j.out
#SBATCH --error=logs/dl_multi_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps

mkdir -p logs

echo "=== Downloading multilingual data ==="
python scripts/download_multilingual.py -o data/multilingual-raw --tokens-per-lang 2500000000
echo "=== Done ==="
