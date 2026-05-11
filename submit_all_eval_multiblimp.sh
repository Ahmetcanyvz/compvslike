#!/usr/bin/env bash
# Submit MultiBLiMP eval jobs for all 4 corrected 1B multilingual models.

set -euo pipefail

MODELS=(
    "bpe_count-multi-128k"
    "compmax-multi-128k"
    "greedyll-exact-multi-128k"
    "unigramlm-multi-128k"
)

mkdir -p logs

for tok in "${MODELS[@]}"; do
    echo "Submitting MultiBLiMP eval: $tok"
    sbatch run_sbatch_eval_multiblimp.sh "$tok"
done

echo ""
echo "All 4 MultiBLiMP eval jobs submitted. Check: squeue -u $USER"
