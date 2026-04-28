#!/usr/bin/env bash
# Submit all 6 corrected 1B training jobs.
# Each job copies the clean checkpoint to outputs_correct/ then resumes with the
# fixed DataModule (manual DistributedSampler, SkipBatchSampler skip survives).

set -euo pipefail

# format: <TOK_NAME> <RESUME_STEP> <DATA_VARIANT>
JOBS=(
    "bpe_count-multi-128k 30000 multi"
    "compmax-multi-128k 30000 multi"
    "greedyll-exact-multi-128k 30000 multi"
    "unigramlm-multi-128k 30000 multi"
    "unigramlm-128k 30000 english"
    "greedyll-exact-128k 20000 english"
)

mkdir -p logs

for job in "${JOBS[@]}"; do
    read -r tok step variant <<<"$job"
    echo "Submitting: $tok from step$step ($variant)"
    sbatch run_sbatch_train_correct.sh "$tok" "$step" "$variant"
done

echo ""
echo "All 6 jobs submitted. Check status with: squeue -u $USER"
