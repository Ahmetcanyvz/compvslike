#!/usr/bin/env bash
# MultiBLiMP + ZhoBLiMP for one multilingual model.
# MultiBLiMP has no cmn config, so Chinese is evaluated with ZhoBLiMP instead.
# Mirrors run_sbatch_eval_multiblimp.sh.
#
#   ./evaluation/run_multiblimp.sh bpe-multi-128k [SEED] [MODEL_SIZE]
set -euo pipefail
source "$(dirname "$0")/../env.sh"

TOK_NAME="${1:?missing TOK_NAME (e.g. bpe-multi-128k)}"
SEED="${2:-42}"
MODEL_SIZE="${3:-me1B-tied}"
BUDGET="${BUDGET:-20}"

MODEL="${MODEL_SIZE}_${TOK_NAME}_${BUDGET}Btok_seed${SEED}"
checkpoint="${CVL_OUTPUTS}/${MODEL}/.checkpoints/last.ckpt"
tokenizer="${CVL_TOKENIZERS}/${TOK_NAME}"
out_dir="${CVL_EVAL_OUT}/${MODEL}"

[[ -f "$checkpoint" ]] || { echo "[ERROR] Checkpoint not found: $checkpoint"; exit 1; }
mkdir -p "$out_dir" "$CVL_LOGS"
echo "=== MultiBLiMP eval: $MODEL ==="

for lang in eng deu spa tur; do
    out_file="$out_dir/multiblimp_${lang}.parquet"
    if [[ -f "$out_file" ]]; then
        echo "  [SKIP] $lang done"
    else
        echo "  [RUN] $lang..."
        python -m src.eval_multiblimp "$checkpoint" "$tokenizer" "$lang" -o "$out_file"
    fi
done

# ZhoBLiMP for Chinese (MultiBLiMP has no cmn config).
zho_out="$out_dir/zhoblimp.parquet"
if [[ -f "$zho_out" ]]; then
    echo "  [SKIP] zhoblimp done"
else
    echo "  [RUN] zhoblimp (cmn)..."
    python -m src.eval_zhoblimp "$checkpoint" "$tokenizer" -o "$zho_out"
fi
echo "=== Done -> ${out_dir} ==="
