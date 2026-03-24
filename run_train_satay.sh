#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Train 100M/340M/500M models: 4 tokenizer types × 3 vocab sizes × 3 seeds
# Machine: satay — 4x RTX 6000 (48GB VRAM)
# Runs 4 tokenizer types in parallel (one per GPU)
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENIZER_BASE="/local/home/ayavuz/compvslike/tokenizers"
DATA_BASE="/local/home/ayavuz/compvslike/data"

MODELS=("me100M-tied" "me340M-tied" "me500M-tied")
VOCAB_SIZES=("128k" "32k" "8k")
SEEDS=(42 43 44)
TOK_TYPES=("bpe" "compmax" "greedyll-exact" "unigramlm")

# ── Helper: generate config and launch training on a GPU ──────────────────────
launch_training() {
    local gpu=$1
    local model=$2
    local tok_type=$3
    local vocab=$4
    local seed=$5

    local tok_name="${tok_type}-${vocab}"
    local config_file="configs/${model}_${tok_name}_seed${seed}.yaml"

    # Set max_tokens based on model size (~20x params)
    local max_tokens
    case "$model" in
        me100M-tied) max_tokens=2_000_000_000 ;;
        me340M-tied) max_tokens=7_000_000_000 ;;
        me500M-tied) max_tokens=10_000_000_000 ;;
        *)           max_tokens=2_000_000_000 ;;
    esac

    cat > "$config_file" <<EOF
paths:
  tokenizer: ${TOKENIZER_BASE}/${tok_name}
  train_data: ${DATA_BASE}/fineweb-edu-${tok_name}/train
  val_data: ${DATA_BASE}/fineweb-edu-${tok_name}/val
  test_data: ${DATA_BASE}/fineweb-edu-${tok_name}/test
  output_dir: ./outputs

model:
  config_path: models/${model}.yaml
  use_flash_attention: true
  use_liger_kernel: true
  torch_compile: false

training:
  seed: ${seed}
  max_tokens: ${max_tokens}
  batch_size: 16
  eval_batch_size: 4
  gradient_accumulation: 8
  sequence_length: 2048
  learning_rate: 6.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  max_grad_norm: 1.0
  warmup_steps: 2000
  decay_steps: 2000
  min_lr_ratio: 0.01
  z_loss_weight: 0

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

    # Check for existing checkpoint to resume from
    local max_tokens_b=$((max_tokens / 1000000000))
    local output_name="${model}_${tok_name}_${max_tokens_b}Btok_seed${seed}"
    local ckpt_dir="outputs/${output_name}/.checkpoints"
    local resume_flag=""
    if [[ -d "$ckpt_dir" ]]; then
        local last_ckpt=$(ls -t "$ckpt_dir"/step*.ckpt 2>/dev/null | head -1)
        if [[ -n "$last_ckpt" ]]; then
            resume_flag="--resume $last_ckpt"
            echo "[GPU ${gpu}] Resuming ${model} / ${tok_name} from ${last_ckpt}"
        fi
    fi

    echo "[GPU ${gpu}] Starting: ${model} / ${tok_name} / seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m src.train "$config_file" --seed "$seed" $resume_flag \
        > "logs/${model}_${tok_name}_seed${seed}.log" 2>&1
    echo "[GPU ${gpu}] Done: ${model} / ${tok_name} / seed${seed}"
}

# ── Main training loop ────────────────────────────────────────────────────────
mkdir -p logs

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# Seed: ${seed}"
    echo "################################################################"

    for vocab in "${VOCAB_SIZES[@]}"; do
        for model in "${MODELS[@]}"; do
            echo ""
            echo "============================================================"
            echo "  ${model} | vocab=${vocab} | seed=${seed} | 4 tokenizers in parallel"
            echo "============================================================"

            # Launch 4 tokenizer types in parallel, one per GPU
            pids=()
            for i in "${!TOK_TYPES[@]}"; do
                launch_training "$i" "$model" "${TOK_TYPES[$i]}" "$vocab" "$seed" &
                pids+=($!)
            done

            # Wait for all 4 to finish
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "[WARN] Process $pid failed"
            done

            echo "  Finished: ${model} / *-${vocab} / seed${seed}"
        done
    done
done

echo ""
echo "=== All training complete ==="
