#!/usr/bin/env bash
# BLiMP evaluation for every trained model. One model per GPU, CVL_NGPU in parallel.
# Set TASKS to restrict to a subset; MULTI=1 runs MultiBLiMP instead.
#
#   ./evaluation/run_blimp.sh
#   MULTI=1 ./evaluation/run_blimp.sh     # MultiBLiMP
#   ZHO=1   ./evaluation/run_blimp.sh     # ZhoBLiMP
set -uo pipefail
source "$(dirname "$0")/../env.sh"

MODULE="src.eval_blimp"; NAME="blimp"
[[ "${MULTI:-0}" == "1" ]] && { MODULE="src.eval_multiblimp"; NAME="multiblimp"; }
[[ "${ZHO:-0}"   == "1" ]] && { MODULE="src.eval_zhoblimp";   NAME="zhoblimp";   }
mkdir -p "$CVL_EVAL_OUT" "$CVL_LOGS"

mapfile -t MODELS < <(ls -d "${CVL_OUTPUTS}"/me*-tied_*seed* 2>/dev/null | xargs -r -n1 basename | sort -u)
(( ${#MODELS[@]} == 0 )) && { echo "No models under ${CVL_OUTPUTS}"; exit 1; }

launched=0 skipped=0
for model in "${MODELS[@]}"; do
    ckpt="${CVL_OUTPUTS}/${model}/.checkpoints/last.ckpt"
    tok_name=$(sed 's/me[0-9]*M-tied_//; s/_[0-9]*Btok_seed[0-9]*//' <<<"$model")
    tokenizer="${CVL_TOKENIZERS}/${tok_name}"
    out="${CVL_EVAL_OUT}/${model}/${NAME}.parquet"
    mkdir -p "$(dirname "$out")"

    if   [[ -f "$out"         ]]; then echo "[skip] ${model}: already evaluated"; ((skipped++)); continue
    elif [[ ! -f "$ckpt"      ]]; then echo "[skip] ${model}: no checkpoint";     ((skipped++)); continue
    elif [[ ! -d "$tokenizer" ]]; then echo "[skip] ${model}: no tokenizer ${tok_name}"; ((skipped++)); continue
    fi

    gpu=$((launched % CVL_NGPU))
    echo "[GPU ${gpu}] ${NAME} ${model}"
    CUDA_VISIBLE_DEVICES=$gpu python -m "$MODULE" \
        "$ckpt" "$tokenizer" -o "$out" ${TASKS:+--tasks "$TASKS"} \
        > "${CVL_LOGS}/${NAME}_${model}.log" 2>&1 &

    ((launched++))
    (( launched % CVL_NGPU == 0 )) && wait
done
wait
echo "=== ${NAME} done: launched=${launched} skipped=${skipped} -> ${CVL_EVAL_OUT} ==="
