#!/usr/bin/env bash
# SLURM wrapper around training/train_models.sh.
# Configure the account/partition in env.local.sh, then:
#
#   sbatch --export=ALL,MODELS=me1B-tied,VOCABS=128k,SEEDS=42 slurm/train.sbatch
#
#SBATCH --job-name=cvl-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/env.sh"

# --account / --partition / --time come from env.local.sh; pass them at submit time:
#   sbatch --account="$CVL_SLURM_ACCOUNT" --partition="$CVL_SLURM_PARTITION" \
#          --time="$CVL_SLURM_TIME" slurm/train.sbatch

cd "$CVL_ROOT"
if [[ -n "$CVL_CONTAINER_ENV" ]]; then
    srun --environment="$CVL_CONTAINER_ENV" bash training/train_models.sh
else
    bash training/train_models.sh
fi
