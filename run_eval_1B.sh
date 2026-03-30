#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCH_LIB=$(uv run --no-sync python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null) || true
[ -n "${TORCH_LIB:-}" ] && export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

CHECKPOINT="/local/home/ayavuz/compvslike/outputs/me1B-tied_bpe-128k_20Btok_seed42/.checkpoints/last.ckpt"
TOKENIZER="/local/home/ayavuz/compvslike/tokenizers/bpe-128k"
RAW_TEST="/local/home/ayavuz/compvslike/data/fineweb-edu-raw/test"
OUT="eval_results/me1B-tied_bpe-128k_20Btok_seed42"

mkdir -p "$OUT"

echo "=== BLiMP ==="
uv run --no-sync python -m src.eval_blimp "$CHECKPOINT" "$TOKENIZER" -o "$OUT/blimp.parquet"

echo ""
echo "=== BPB ==="
uv run --no-sync python -m src.eval_bpb "$CHECKPOINT" "$TOKENIZER" "$RAW_TEST" -o "$OUT/bpb.parquet"
