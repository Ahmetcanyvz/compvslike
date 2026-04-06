#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCH_LIB=$(uv run --no-sync python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null) || true
[ -n "${TORCH_LIB:-}" ] && export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
RAW_TEST="/local/home/ayavuz/compvslike/data/fineweb-edu-raw/test"
OUTPUTS="/local/home/ayavuz/compvslike/outputs"
EVAL_OUT="eval_results"

MODELS=(
    "me1B-tied_bpe-128k_20Btok_seed42"
    "me1B-tied_compmax-128k_20Btok_seed42"
)

for model in "${MODELS[@]}"; do
    tok_name=$(echo "$model" | sed 's/me1B-tied_//;s/_20Btok_seed42//')
    checkpoint="${OUTPUTS}/${model}/.checkpoints/last.ckpt"
    tokenizer="${TOKENIZER_BASE}/${tok_name}"
    out_dir="${EVAL_OUT}/${model}"

    if [[ ! -f "$checkpoint" ]]; then
        echo "[SKIP] $checkpoint not found"
        continue
    fi

    mkdir -p "$out_dir"
    echo "=== $model (tokenizer: $tok_name) ==="

    if [[ -f "$out_dir/blimp.parquet" ]]; then
        echo "  [SKIP] BLiMP done"
    else
        echo "  [RUN] BLiMP..."
        uv run --no-sync python -m src.eval_blimp "$checkpoint" "$tokenizer" -o "$out_dir/blimp.parquet" || echo "  [FAIL] BLiMP"
    fi

    if [[ -f "$out_dir/bpb.parquet" ]]; then
        echo "  [SKIP] BPB done"
    else
        echo "  [RUN] BPB..."
        uv run --no-sync python -m src.eval_bpb "$checkpoint" "$tokenizer" "$RAW_TEST" -o "$out_dir/bpb.parquet" || echo "  [FAIL] BPB"
    fi
done

echo "=== Done ==="
