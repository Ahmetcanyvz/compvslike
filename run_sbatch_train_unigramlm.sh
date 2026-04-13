#!/usr/bin/env bash
#SBATCH --job-name=train-1B-unigramlm
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_unigramlm_%j.out
#SBATCH --error=logs/train_unigramlm_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
bash run_srun_clariden.sh unigramlm
