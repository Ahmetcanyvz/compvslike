#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch evaluation script: BLiMP + BPB for all trained models
# Evaluates me57M-tied models across 3 seeds x 3 tokenizer types x 3 vocab sizes
# =============================================================================

# ── GPU selection ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Base paths (edit these for your setup) ────────────────────────────────────
TOKENIZER_BASE="/local/home/ayavuz/data_all/tokenizers"
RAW_TEST_DATA="/local/home/ayavuz/data_all/data/fineweb-edu-raw/test"
EVAL_OUTPUT_BASE="eval_results"
CHECKPOINT_NAME=".checkpoints/step10000.ckpt"

# Seed directories (note the inconsistent naming)
SEED_DIRS=(
    "/local/home/ayavuz/outputs_seed_42"
    "/local/home/ayavuz/output_seed_43"
    "/local/home/ayavuz/outputs_seed_44"
)

# ── Main loop ─────────────────────────────────────────────────────────────────
for seed_dir in "${SEED_DIRS[@]}"; do
    if [[ ! -d "$seed_dir" ]]; then
        echo "[WARN] Seed directory not found, skipping: $seed_dir"
        continue
    fi

    echo "=== Processing seed directory: $seed_dir ==="

    for model_dir in "$seed_dir"/me57M-tied_*/; do
        # Strip trailing slash and get folder name
        model_dir="${model_dir%/}"
        folder_name="$(basename "$model_dir")"

        # Check checkpoint exists
        checkpoint="$model_dir/$CHECKPOINT_NAME"
        if [[ ! -f "$checkpoint" ]]; then
            echo "[WARN] Checkpoint not found, skipping: $checkpoint"
            continue
        fi

        # Extract tokenizer name: me57M-tied_bpe-8k_15000steps_seed43 → bpe-8k
        tok_name="${folder_name#me57M-tied_}"          # strip prefix
        tok_name="${tok_name%%_[0-9]*steps_seed*}"     # strip _NNNNNsteps_seedNN suffix

        tokenizer_path="$TOKENIZER_BASE/$tok_name"
        if [[ ! -d "$tokenizer_path" ]]; then
            echo "[WARN] Tokenizer not found, skipping: $tokenizer_path"
            continue
        fi

        # Output directory for this model
        out_dir="$EVAL_OUTPUT_BASE/$folder_name"
        mkdir -p "$out_dir"

        echo "--- Evaluating: $folder_name (tokenizer: $tok_name) ---"

        # ── BLiMP ─────────────────────────────────────────────────────────
        if [[ -f "$out_dir/blimp.parquet" ]]; then
            echo "  [SKIP] BLiMP results already exist: $out_dir/blimp.parquet"
        else
            echo "  [RUN]  BLiMP evaluation..."
            uv run python -m src.eval_blimp \
                "$checkpoint" \
                "$tokenizer_path" \
                -o "$out_dir/blimp.parquet" \
            || echo "  [FAIL] BLiMP evaluation failed for $folder_name"
        fi

        # ── BPB ───────────────────────────────────────────────────────────
        if [[ -f "$out_dir/bpb.parquet" ]]; then
            echo "  [SKIP] BPB results already exist: $out_dir/bpb.parquet"
        else
            echo "  [RUN]  BPB evaluation..."
            uv run python -m src.eval_bpb \
                "$checkpoint" \
                "$tokenizer_path" \
                "$RAW_TEST_DATA" \
                -o "$out_dir/bpb.parquet" \
            || echo "  [FAIL] BPB evaluation failed for $folder_name"
        fi
    done
done

echo ""
echo "=== All evaluations complete. Results in: $EVAL_OUTPUT_BASE/ ==="
