#!/usr/bin/env bash
#SBATCH --job-name=boot-1B
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/boot_1B_%j.out
#SBATCH --error=logs/boot_1B_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

echo "=== Paired bootstrap: 1B English, all pairs ==="
python scripts/bootstrap_bpb_compare.py eval_results_1B \
    --glob "me1B-tied_*-128k_20Btok_seed42/bpb.parquet" \
    --all-pairs \
    -o eval_results_1B/cmp_1B_allpairs.json
echo "=== Done ==="
