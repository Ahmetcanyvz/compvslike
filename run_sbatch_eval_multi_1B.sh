#!/usr/bin/env bash
#SBATCH --job-name=eval-1B-multi
#SBATCH --partition=normal
#SBATCH --account=a139
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/eval_1B_multi_%j.out
#SBATCH --error=logs/eval_1B_multi_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
TOKENIZER_BASE="${WORK_DIR}/tokenizers"
RAW_BASE="${WORK_DIR}/data/multilingual-raw"
ENG_RAW_TEST="${WORK_DIR}/data/fineweb-edu-raw/test"
OUTPUTS="${WORK_DIR}/outputs"
EVAL_OUT="${WORK_DIR}/eval_results"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs "$EVAL_OUT"

MODELS=(
    "me1B-tied_compmax-multi-128k_20Btok_seed42:compmax-multi-128k"
    "me1B-tied_unigramlm-multi-128k_20Btok_seed42:unigramlm-multi-128k"
)

LANGS=(eng deu spa tur cmn)

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    tok_name="${entry##*:}"

    checkpoint="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    tokenizer="${TOKENIZER_BASE}/${tok_name}"
    out_dir="${EVAL_OUT}/${model}"

    if [[ ! -f "$checkpoint" ]]; then
        echo "[SKIP] Checkpoint not found: $checkpoint"
        continue
    fi

    mkdir -p "$out_dir"
    echo "=== $model (tokenizer: $tok_name) ==="

    # BLiMP (English-only)
    if [[ -f "$out_dir/blimp.parquet" ]]; then
        echo "  [SKIP] BLiMP done"
    else
        echo "  [RUN] BLiMP..."
        python -m src.eval_blimp "$checkpoint" "$tokenizer" -o "$out_dir/blimp.parquet"
    fi

    # BPB per language
    for lang in "${LANGS[@]}"; do
        if [[ "$lang" == "eng" ]]; then
            test_path="$ENG_RAW_TEST"
        else
            test_path="${RAW_BASE}/${lang}/test"
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
done

echo "=== Done ==="
