#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
RAW_TEST="/local/home/ayavuz/compvslike/data/fineweb-edu-raw/test"
OUTPUTS="/local/home/ayavuz/compvslike/outputs"
EVAL_OUT="eval_results"

mkdir -p logs

eval_model() {
    local gpu=$1
    local model=$2
    local tok_name=$3

    local checkpoint="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    local tokenizer="${TOKENIZER_BASE}/${tok_name}"
    local out_dir="${EVAL_OUT}/${model}"

    if [[ ! -f "$checkpoint" ]]; then
        echo "[GPU ${gpu}] SKIP - checkpoint not found: $checkpoint"
        return
    fi

    mkdir -p "$out_dir"
    echo "[GPU ${gpu}] Evaluating: $model"

    if [[ ! -f "$out_dir/blimp.parquet" ]]; then
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.eval_blimp "$checkpoint" "$tokenizer" -o "$out_dir/blimp.parquet" 2>&1 | tail -5
    fi

    if [[ ! -f "$out_dir/bpb.parquet" ]]; then
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.eval_bpb "$checkpoint" "$tokenizer" "$RAW_TEST" -o "$out_dir/bpb.parquet" 2>&1 | tail -5
    fi

    echo "[GPU ${gpu}] Done: $model"
}

eval_model 0 "me340M-tied_bpe-128k_7Btok_seed42" "bpe-128k"
eval_model 0 "me340M-tied_compmax-128k_7Btok_seed42" "compmax-128k"
eval_model 0 "me340M-tied_greedyll-exact-128k_7Btok_seed42" "greedyll-exact-128k"
eval_model 0 "me340M-tied_unigramlm-128k_7Btok_seed42" "unigramlm-128k"
echo "=== All done ==="
