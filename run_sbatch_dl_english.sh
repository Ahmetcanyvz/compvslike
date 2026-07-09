#!/usr/bin/env bash
#SBATCH --job-name=dl-eng
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/dl_eng_%j.out
#SBATCH --error=logs/dl_eng_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Downloading English FineWeb-Edu (20B tokens) ==="
python scripts/download_english.py -o data/fineweb-edu-raw --target-tokens 20000000000
echo "=== Done ==="
