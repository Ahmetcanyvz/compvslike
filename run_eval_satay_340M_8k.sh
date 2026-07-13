#!/usr/bin/env bash
# BPB eval for 340M 8k models on satay: one model per GPU, 4 in parallel.
# Bare-metal (uv, no SLURM). Auto-discovers me340M-tied_*-8k_* checkpoints.
set -uo pipefail

BASE="/local/home/ayavuz/compvslike"
TOKENIZER_BASE="${BASE}/tokenizers"
RAW_TEST="${BASE}/data/fineweb-edu-raw/test"   # English test (47,384 docs)
OUTPUTS="${BASE}/outputs"
EVAL_OUT="${BASE}/eval_results_340M_8k"
NGPU=4

cd "$BASE"
mkdir -p "$EVAL_OUT" logs

# Discover trained 340M 8k models (any tokenizer / seed).
mapfile -t MODELS < <(ls -d "${OUTPUTS}"/me340M-tied_*-8k_*seed* 2>/dev/null | xargs -r -n1 basename)
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No me340M-tied_*-8k_* models found under ${OUTPUTS}"
    exit 1
fi
echo "Found ${#MODELS[@]} models to consider."

launched=0
for model in "${MODELS[@]}"; do
    ckpt="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    # tok name: strip prefix and _<N>Btok_seed<seed> suffix -> e.g. bpe-8k
    tok_name=$(echo "$model" | sed 's/me340M-tied_//; s/_[0-9]*Btok_seed[0-9]*//')
    tokenizer="${TOKENIZER_BASE}/${tok_name}"
    out_dir="${EVAL_OUT}/${model}"
    out="${out_dir}/bpb.parquet"
    mkdir -p "$out_dir"

    if [[ -f "$out" ]]; then
        echo "[skip] ${model}: bpb.parquet exists"
        continue
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] ${model}: checkpoint missing ($ckpt)"
        continue
    fi
    if [[ ! -d "$tokenizer" ]]; then
        echo "[skip] ${model}: tokenizer missing ($tokenizer)"
        continue
    fi

    gpu=$((launched % NGPU))
    echo "[GPU ${gpu}] eval ${model} (tok ${tok_name})"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.eval_bpb \
        "$ckpt" "$tokenizer" "$RAW_TEST" -o "$out" \
        > "logs/eval_${model}.log" 2>&1 &

    launched=$((launched + 1))
    # After filling all GPUs, wait for the batch to finish before the next 4.
    if (( launched % NGPU == 0 )); then
        wait
    fi
done

wait  # let the final (partial) batch finish
echo "=== All 340M 8k BPB evals done -> ${EVAL_OUT}/ ==="
