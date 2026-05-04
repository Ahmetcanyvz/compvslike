#!/usr/bin/env bash
#SBATCH --job-name=exploration
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/exploration_%j.out
#SBATCH --error=logs/exploration_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_exploration.sh <VARIANT>
#   VARIANT = "english" or "multi"
# Example:
#   sbatch run_sbatch_exploration.sh english
#   sbatch run_sbatch_exploration.sh multi

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
VARIANT="${1:-english}"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs

OUT_DIR="exploration_results/${VARIANT}"
mkdir -p "$OUT_DIR"

export LM_TRAINER_ROOT="$WORK_DIR"
export EXPLORATION_VARIANT="$VARIANT"

cd scripts/exploration

echo "=== Exploration variant: ${VARIANT} ==="
echo "=== LM_TRAINER_ROOT: ${LM_TRAINER_ROOT} ==="
echo ""

run_step() {
    local name=$1
    shift
    local logfile="${WORK_DIR}/${OUT_DIR}/${name}.log"
    echo "[RUN] ${name} -> ${logfile}"
    if python "$@" > "$logfile" 2>&1; then
        echo "[OK]  ${name}"
    else
        echo "[FAIL] ${name}  (see ${logfile})"
    fi
}

run_step compression_stats     compression_stats.py
run_step token_length_dist     token_length_dist.py
run_step zipf_and_entropy      zipf_and_entropy.py
run_step vocab_differences     vocab_differences.py
run_step merge_overlap         merge_overlap.py
run_step kill_ratio            kill_ratio.py
run_step tokenization_examples tokenization_examples.py

echo ""
echo "=== Done. Results in ${OUT_DIR}/ ==="
