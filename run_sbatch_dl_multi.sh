#!/usr/bin/env bash
#SBATCH --job-name=dl-multi
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3
#SBATCH --output=logs/dl_multi_%A_%a.out
#SBATCH --error=logs/dl_multi_%A_%a.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

LANGS=(deu spa tur cmn)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

echo "=== Downloading multilingual: ${LANG} (2.5B tokens) ==="
python scripts/download_multilingual.py -o data/multilingual-raw --tokens-per-lang 2500000000 --lang "$LANG"
echo "=== Done: ${LANG} ==="
