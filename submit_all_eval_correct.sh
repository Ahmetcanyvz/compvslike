#!/usr/bin/env bash
# Submit eval jobs for the 5 corrected 1B models (greedyll-exact-128k still training).

set -euo pipefail

# format: <TOK_NAME> <DATA_VARIANT>
JOBS=(
    "bpe_count-multi-128k multi"
    "compmax-multi-128k multi"
    "greedyll-exact-multi-128k multi"
    "unigramlm-multi-128k multi"
    "unigramlm-128k english"
)

mkdir -p logs

for job in "${JOBS[@]}"; do
    read -r tok variant <<<"$job"
    echo "Submitting eval: $tok ($variant)"
    sbatch run_sbatch_eval_correct.sh "$tok" "$variant"
done

echo ""
echo "All 5 eval jobs submitted. Check status with: squeue -u $USER"
