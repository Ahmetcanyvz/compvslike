#!/usr/bin/env bash
#SBATCH --job-name=tok-eng-arr
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3
#SBATCH --output=logs/tok_eng_%A_%a.out
#SBATCH --error=logs/tok_eng_%A_%a.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

TOKS=(bpe_count-128k compmax-128k greedyll-exact-128k unigramlm-128k)
TOK=${TOKS[$SLURM_ARRAY_TASK_ID]}

# One tokenizer per node, all 4 in parallel, reading the shared raw data.
# --tokenize-only: raw is already downloaded; just tokenize (train+train_extra+val+test).
# num_proc kept at 64 (not 288) to stay well under memory — 288 OOMs.
echo "=== Tokenizing ${TOK} (tokenize-only, num_proc=64) ==="
python scripts/prepare_all.py \
    -t tokenizers/${TOK} \
    -o data \
    --raw-data-dir data/fineweb-edu-raw \
    --tokenize-only \
    --num-proc 64
echo "=== Done: ${TOK} ==="
