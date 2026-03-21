#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Train me1B-tied models: 4 tokenizer types × 128k vocab × seed 42
# Machine: 1x RTX Pro 6000 (96GB VRAM)
# =============================================================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Configuration ─────────────────────────────────────────────────────────────
TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
DATA_BASE="/local/home/ayavuz/compvslike/data"
SEED=42

TOKENIZERS=(
    "bpe-128k"
    "compmax-128k"
    "greedyll-exact-128k"
    "unigramlm-128k"
)

# ── Training loop ─────────────────────────────────────────────────────────────
for tok in "${TOKENIZERS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Training: me1B-tied with ${tok} (seed ${SEED})"
    echo "============================================================"

    CONFIG_FILE="configs/train_1B_${tok}.yaml"

    # Generate config
    cat > "$CONFIG_FILE" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${tok}
  train_data: ${DATA_BASE}/fineweb-edu-${tok}/train
  val_data: ${DATA_BASE}/fineweb-edu-${tok}/val
  test_data: ${DATA_BASE}/fineweb-edu-${tok}/test
  output_dir: ./outputs

model:
  config_path: models/me1B-tied.yaml
  use_flash_attention: true
  use_liger_kernel: false
  torch_compile: false

training:
  seed: ${SEED}
  max_tokens: 20_000_000_000
  batch_size: 8
  gradient_accumulation: 16
  sequence_length: 2048
  learning_rate: 6.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  warmup_steps: 2000
  decay_steps: 2000
  min_lr_ratio: 0.01
  z_loss_weight: 1.0e-4

checkpoint:
  save_every_n_steps: 5000
  save_dir: .checkpoints
  save_last: true

hardware:
  accelerator: gpu
  devices: 1
  precision: bf16-true
  strategy: auto

logging:
  log_every_n_steps: 50
  log_loss_every_n_steps: 1000
EOF

    echo "Config written to ${CONFIG_FILE}"

    # Run training
    uv run python -m src.train train "$CONFIG_FILE" --seed "$SEED" \
        || echo "[FAIL] Training failed for ${tok}"

    echo "Finished: ${tok}"
done

echo ""
echo "=== All 1B trainings complete ==="
