#!/usr/bin/env bash
#SBATCH --job-name=eval-correct
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/eval_correct_%j.out
#SBATCH --error=logs/eval_correct_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_eval_correct.sh <TOK_NAME> <DATA_VARIANT>
# DATA_VARIANT = "multi" (per-language BPB on eng/deu/spa/tur/cmn) or "english" (BPB on fineweb-edu test only)
# Examples:
#   sbatch run_sbatch_eval_correct.sh bpe_count-multi-128k multi
#   sbatch run_sbatch_eval_correct.sh unigramlm-128k english

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
RAW_MULTI_BASE="${WORK_DIR}/data/multilingual-raw"
ENG_RAW_TEST="${WORK_DIR}/data/fineweb-edu-raw/test"
OUTPUTS="${WORK_DIR}/outputs_correct"
EVAL_OUT="${WORK_DIR}/eval_results_correct"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs "$EVAL_OUT"

TOK_NAME="${1:?missing TOK_NAME}"
DATA_VARIANT="${2:?missing DATA_VARIANT (multi or english)}"
SEED=42

MODEL="me1B-tied_${TOK_NAME}_20Btok_seed${SEED}"
checkpoint="${OUTPUTS}/${MODEL}/.checkpoints/last.ckpt"
tokenizer="${TOKENIZER_BASE}/${TOK_NAME}"
out_dir="${EVAL_OUT}/${MODEL}"

if [[ ! -f "$checkpoint" ]]; then
    echo "[ERROR] Checkpoint not found: $checkpoint"
    exit 1
fi

mkdir -p "$out_dir"
echo "=== Evaluating: $MODEL (tokenizer: $TOK_NAME, variant: $DATA_VARIANT) ==="

# BLiMP (English-only, applicable to all models)
if [[ -f "$out_dir/blimp.parquet" ]]; then
    echo "  [SKIP] BLiMP done"
else
    echo "  [RUN] BLiMP..."
    python -m src.eval_blimp "$checkpoint" "$tokenizer" -o "$out_dir/blimp.parquet"
fi

if [[ "$DATA_VARIANT" == "multi" ]]; then
    # BPB per language for multilingual models
    LANGS=(eng deu spa tur cmn)
    for lang in "${LANGS[@]}"; do
        if [[ "$lang" == "eng" ]]; then
            test_path="$ENG_RAW_TEST"
        else
            test_path="${RAW_MULTI_BASE}/${lang}/test"
        fi

        if [[ ! -d "$test_path" ]]; then
            echo "  [SKIP] $lang test path not found: $test_path"
            continue
        fi

        out_file="$out_dir/bpb_${lang}.parquet"
        if [[ -f "$out_file" ]]; then
            echo "  [SKIP] BPB $lang done"
        else
            echo "  [RUN] BPB $lang..."
            python -m src.eval_bpb "$checkpoint" "$tokenizer" "$test_path" -o "$out_file"
        fi
    done
elif [[ "$DATA_VARIANT" == "english" ]]; then
    # BPB on English test only
    out_file="$out_dir/bpb.parquet"
    if [[ -f "$out_file" ]]; then
        echo "  [SKIP] BPB done"
    else
        echo "  [RUN] BPB..."
        python -m src.eval_bpb "$checkpoint" "$tokenizer" "$ENG_RAW_TEST" -o "$out_file"
    fi
else
    echo "[ERROR] DATA_VARIANT must be 'multi' or 'english'"
    exit 1
fi

echo "=== Done: $MODEL ==="
