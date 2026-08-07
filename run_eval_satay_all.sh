#!/usr/bin/env bash
# BPB eval for ALL me*-tied models under outputs/ on satay: one model per GPU,
# 4 in parallel. Bare-metal (uv, no SLURM). Auto-discovers every model with a
# checkpoint; skips any that already has a bpb.parquet.
set -uo pipefail

BASE="/local/home/ayavuz/compvslike"
TOKENIZER_BASE="${BASE}/tokenizers"
RAW_TEST="${BASE}/data/fineweb-edu-raw/test"   # English test (47,384 docs)
OUTPUTS="${BASE}/outputs"
EVAL_OUT="${BASE}/eval_results_all"
NGPU=4

cd "$BASE"
mkdir -p "$EVAL_OUT" logs

# Discover every me<size>M-tied_<tok>-<vocab>_<N>Btok_seed<seed> model.
mapfile -t MODELS < <(ls -d "${OUTPUTS}"/me*-tied_*seed* 2>/dev/null | xargs -r -n1 basename | sort -u)
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No me*-tied models found under ${OUTPUTS}"
    exit 1
fi
echo "Found ${#MODELS[@]} models to consider."

launched=0
done_cnt=0
skip_cnt=0
for model in "${MODELS[@]}"; do
    ckpt="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    # tok name: strip me<size>M-tied_ prefix and _<N>Btok_seed<seed> suffix -> e.g. bpe-8k / compmax-128k
    tok_name=$(echo "$model" | sed 's/me[0-9]*M-tied_//; s/_[0-9]*Btok_seed[0-9]*//')
    tokenizer="${TOKENIZER_BASE}/${tok_name}"
    out_dir="${EVAL_OUT}/${model}"
    out="${out_dir}/bpb.parquet"
    mkdir -p "$out_dir"

    if [[ -f "$out" ]]; then
        echo "[skip] ${model}: bpb.parquet exists"; skip_cnt=$((skip_cnt+1)); continue
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "[skip] ${model}: checkpoint missing"; skip_cnt=$((skip_cnt+1)); continue
    fi
    if [[ ! -d "$tokenizer" ]]; then
        echo "[skip] ${model}: tokenizer missing ($tokenizer)"; skip_cnt=$((skip_cnt+1)); continue
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
echo "=== Done. launched=${launched}, skipped=${skip_cnt}. Results -> ${EVAL_OUT}/ ==="
