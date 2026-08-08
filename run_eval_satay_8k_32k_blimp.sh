#!/usr/bin/env bash
# BLiMP eval for 340M/500M models at 8k and 32k vocab on satay: one model per
# GPU, 4 in parallel. Writes blimp.parquet next to bpb.parquet in the same dir.
set -uo pipefail

BASE="/local/home/ayavuz/compvslike"
TOKENIZER_BASE="${BASE}/tokenizers"
OUTPUTS="${BASE}/outputs"
EVAL_OUT="${BASE}/eval_results_8k_32k"
NGPU=4

cd "$BASE"
mkdir -p "$EVAL_OUT" logs

mapfile -t MODELS < <(
    ls -d "${OUTPUTS}"/me340M-tied_*-8k_*seed* \
          "${OUTPUTS}"/me340M-tied_*-32k_*seed* \
          "${OUTPUTS}"/me500M-tied_*-8k_*seed* \
          "${OUTPUTS}"/me500M-tied_*-32k_*seed* 2>/dev/null | xargs -r -n1 basename | sort -u
)
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No me340M/500M-tied 8k/32k models found under ${OUTPUTS}"
    exit 1
fi
echo "Found ${#MODELS[@]} models to consider."

launched=0
for model in "${MODELS[@]}"; do
    ckpt="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    tok_name=$(echo "$model" | sed 's/me[0-9]*M-tied_//; s/_[0-9]*Btok_seed[0-9]*//')
    tokenizer="${TOKENIZER_BASE}/${tok_name}"
    out_dir="${EVAL_OUT}/${model}"
    out="${out_dir}/blimp.parquet"
    mkdir -p "$out_dir"

    if [[ -f "$out" ]]; then
        echo "[skip] ${model}: blimp.parquet exists"; continue
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] ${model}: checkpoint missing"; continue
    fi
    if [[ ! -d "$tokenizer" ]]; then
        echo "[skip] ${model}: tokenizer missing ($tokenizer)"; continue
    fi

    gpu=$((launched % NGPU))
    echo "[GPU ${gpu}] BLiMP ${model} (tok ${tok_name})"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.eval_blimp \
        "$ckpt" "$tokenizer" -o "$out" \
        > "logs/blimp_${model}.log" 2>&1 &

    launched=$((launched + 1))
    if (( launched % NGPU == 0 )); then
        wait
    fi
done

wait
echo "=== All 8k/32k BLiMP evals done -> ${EVAL_OUT}/ ==="
