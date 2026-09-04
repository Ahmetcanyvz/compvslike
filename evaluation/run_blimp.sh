#!/usr/bin/env bash
# BLiMP (English) for every trained model. One model per GPU, CVL_NGPU in parallel.
# Mirrors run_eval_satay_8k_32k_blimp.sh.
#
#   ./evaluation/run_blimp.sh
set -uo pipefail
source "$(dirname "$0")/../env.sh"
mkdir -p "$CVL_EVAL_OUT" "$CVL_LOGS"

mapfile -t MODELS < <(ls -d "${CVL_OUTPUTS}"/me*-tied_*seed* 2>/dev/null | xargs -r -n1 basename | sort -u)
(( ${#MODELS[@]} == 0 )) && { echo "No models under ${CVL_OUTPUTS}"; exit 1; }

launched=0 skipped=0
for model in "${MODELS[@]}"; do
    ckpt="${CVL_OUTPUTS}/${model}/.checkpoints/last.ckpt"
    tok_name=$(sed 's/me[0-9]*M-tied_//; s/_[0-9]*Btok_seed[0-9]*//' <<<"$model")
    tokenizer="${CVL_TOKENIZERS}/${tok_name}"
    out="${CVL_EVAL_OUT}/${model}/blimp.parquet"
    mkdir -p "$(dirname "$out")"

    if   [[ -f "$out"         ]]; then echo "[skip] ${model}: already evaluated"; ((skipped++)); continue
    elif [[ ! -f "$ckpt"      ]]; then echo "[skip] ${model}: no checkpoint";     ((skipped++)); continue
    elif [[ ! -d "$tokenizer" ]]; then echo "[skip] ${model}: no tokenizer ${tok_name}"; ((skipped++)); continue
    fi

    gpu=$((launched % CVL_NGPU))
    echo "[GPU ${gpu}] BLiMP ${model} (tok ${tok_name})"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.eval_blimp \
        "$ckpt" "$tokenizer" -o "$out" \
        > "${CVL_LOGS}/blimp_${model}.log" 2>&1 &

    ((launched++))
    (( launched % CVL_NGPU == 0 )) && wait
done
wait
echo "=== blimp done: launched=${launched} skipped=${skipped} -> ${CVL_EVAL_OUT} ==="
