#!/usr/bin/env bash
#SBATCH --job-name=dl-all
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/dl_all_%j.out
#SBATCH --error=logs/dl_all_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Downloading English FineWeb-Edu (20B tokens) ==="
python scripts/download_english.py -o data/fineweb-edu-raw --target-tokens 20000000000

echo "=== Downloading multilingual (2.5B per lang, 4 languages) ==="
python scripts/download_multilingual.py -o data/multilingual-raw --tokens-per-lang 2500000000

echo "=== Done ==="
