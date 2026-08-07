#!/usr/bin/env bash
# BPB eval for 340M/500M models at 8k and 32k vocab on satay: one model per GPU,
# 4 in parallel. Bare-metal (uv, no SLURM). Auto-discovers all matching models.
set -uo pipefail

BASE="/local/home/ayavuz/compvslike"
TOKENIZER_BASE="${BASE}/tokenizers"
RAW_TEST="${BASE}/data/fineweb-edu-raw/test"   # English test (47,384 docs)
OUTPUTS="${BASE}/outputs"
EVAL_OUT="${BASE}/eval_results_8k_32k"
NGPU=4

cd "$BASE"
mkdir -p "$EVAL_OUT" logs

# Discover 340M/500M models at 8k or 32k vocab (any tokenizer / seed).
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
    # tok name: strip me<size>M-tied_ prefix and _<N>Btok_seed<seed> suffix -> e.g. bpe-8k / compmax-32k
    tok_name=$(echo "$model" | sed 's/me[0-9]*M-tied_//; s/_[0-9]*Btok_seed[0-9]*//')
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
    if (( launched % NGPU == 0 )); then
        wait
    fi
done

wait
echo "=== All 8k/32k BPB evals done -> ${EVAL_OUT}/ ==="
