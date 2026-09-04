#!/usr/bin/env bash
# Convenience submitter: fills SLURM flags from env.local.sh.
#   ./slurm/submit_train.sh MODELS=me1B-tied VOCABS=128k SEEDS=42
set -euo pipefail
source "$(dirname "$0")/../env.sh"
[[ "$CVL_SLURM_ACCOUNT" == "CHANGE_ME" ]] && { echo "Set CVL_SLURM_ACCOUNT in env.local.sh first."; exit 1; }
mkdir -p "$CVL_LOGS"
exec sbatch \
    --account="$CVL_SLURM_ACCOUNT" \
    --partition="$CVL_SLURM_PARTITION" \
    --time="$CVL_SLURM_TIME" \
    --export=ALL${1:+,$(IFS=,; echo "$*")} \
    "$CVL_ROOT/slurm/train.sbatch"
