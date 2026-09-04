#!/usr/bin/env bash
# Submits the training array, filling SLURM flags from env.local.sh.
#   ./slurm/submit.sh                                  # seeds 43,44 x 4 tokenizers
#   MODEL=me1B-tied SEEDS="42" ./slurm/submit.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"
[[ "$CVL_SLURM_ACCOUNT" == "CHANGE_ME" ]] && { echo "Set CVL_SLURM_ACCOUNT in env.local.sh first."; exit 1; }

read -r -a _seeds <<< "${SEEDS:-43 44}"
read -r -a _toks  <<< "${TOKS:-bpe-128k topdowncomp-128k bottomupll-exact-128k unigramlm-128k}"
n=$(( ${#_seeds[@]} * ${#_toks[@]} ))
mkdir -p "${CVL_ROOT}/logs_training"

exec sbatch \
    --account="$CVL_SLURM_ACCOUNT" \
    --partition="$CVL_SLURM_PARTITION" \
    --time="$CVL_SLURM_TIME" \
    --array=0-$((n - 1)) \
    --export=ALL,MODEL="${MODEL:-me1B-tied}",SEEDS="${SEEDS:-43 44}",TOKS="${TOKS:-bpe-128k topdowncomp-128k bottomupll-exact-128k unigramlm-128k}",BUDGET_B="${BUDGET_B:-20}" \
    "$CVL_ROOT/slurm/train_array.sbatch"
