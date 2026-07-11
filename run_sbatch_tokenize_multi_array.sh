#!/usr/bin/env bash
#SBATCH --job-name=tok-multi-arr
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --array=0-3
#SBATCH --output=logs/tok_multi_%A_%a.out
#SBATCH --error=logs/tok_multi_%A_%a.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps
mkdir -p logs

TOKS=(bpe_count-multi-128k compmax-multi-128k greedyll-exact-multi-128k unigramlm-multi-128k)
TOK=${TOKS[$SLURM_ARRAY_TASK_ID]}

# One tokenizer per node, all 4 in parallel, reading the shared merged data.
# --tokenize-only: merge already built by run_sbatch_merge_multi.sh; just tokenize.
echo "=== Tokenizing ${TOK} (tokenize-only, num_proc=64) ==="
python scripts/prepare_multilingual.py \
    -t tokenizers/${TOK} \
    -o data/multilingual \
    --raw-data-dir data/multilingual-raw \
    --total-tokens 20000000000 \
    --eng-raw-dir data/fineweb-edu-raw \
    --tokenize-only \
    --num-proc 64
echo "=== Done: ${TOK} ==="
