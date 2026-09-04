#!/usr/bin/env bash
# Train the paper's LM grid: tokenizer types x vocab sizes x seeds.
# One model per GPU, CVL_NGPU in parallel. Resumes from last checkpoint if present.
#
#   ./training/train_models.sh                                  # defaults below
#   MODELS="me340M-tied" VOCABS="8k 32k" SEEDS="42" ./training/train_models.sh
set -uo pipefail
source "$(dirname "$0")/../env.sh"

MODELS="${MODELS:-me340M-tied}"
VOCABS="${VOCABS:-128k 32k 8k}"
SEEDS="${SEEDS:-42 43 44}"
TOK_TYPES="${TOK_TYPES:-bpe compmax greedyll-exact unigramlm}"

mkdir -p "$CVL_LOGS" "$CVL_ROOT/configs/generated"

# Token budget ~= 20x parameters (Chinchilla-style), as used in the paper.
token_budget() {
    case "$1" in
        me100M-tied)  echo 2000000000  ;;
        me340M-tied)  echo 7000000000  ;;
        me500M-tied)  echo 10000000000 ;;
        me1B-tied)    echo 20000000000 ;;
        *)            echo 2000000000  ;;
    esac
}

launch() {
    local gpu=$1 model=$2 tok_type=$3 vocab=$4 seed=$5
    local tok_name="${tok_type}-${vocab}"
    local max_tokens; max_tokens=$(token_budget "$model")
    local cfg="$CVL_ROOT/configs/generated/${model}_${tok_name}_seed${seed}.yaml"

    local batch_size=16 grad_accum=8 eval_bs=4
    [[ "$model" == "me100M-tied" ]] && { batch_size=32; grad_accum=4; }

    cat > "$cfg" <<EOF
paths:
  tokenizer: ${CVL_TOKENIZERS}/${tok_name}
  train_data: ${CVL_DATA}/fineweb-edu-${tok_name}/train
  val_data: ${CVL_DATA}/fineweb-edu-${tok_name}/val
  test_data: ${CVL_DATA}/fineweb-edu-${tok_name}/test
  output_dir: ${CVL_OUTPUTS}

model:
  config_path: ${CVL_ROOT}/configs/models/${model}.yaml
  use_flash_attention: true
  use_liger_kernel: true
  torch_compile: false

training:
  seed: ${seed}
  max_tokens: ${max_tokens}
  batch_size: ${batch_size}
  eval_batch_size: ${eval_bs}
  gradient_accumulation: ${grad_accum}
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

    local out_name="${model}_${tok_name}_$((max_tokens / 1000000000))Btok_seed${seed}"
    local ckpt_dir="${CVL_OUTPUTS}/${out_name}/.checkpoints"
    local resume=""
    if [[ -d "$ckpt_dir" ]]; then
        local last; last=$(ls -t "$ckpt_dir"/step*.ckpt 2>/dev/null | head -1)
        [[ -n "$last" ]] && { resume="--resume $last"; echo "[GPU ${gpu}] resuming ${out_name} from $(basename "$last")"; }
    fi

    echo "[GPU ${gpu}] start ${out_name}"
    CUDA_VISIBLE_DEVICES=$gpu python -m src.train train "$cfg" --seed "$seed" $resume \
        > "${CVL_LOGS}/${out_name}.log" 2>&1
    echo "[GPU ${gpu}] done  ${out_name}"
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for seed in $SEEDS; do
  for vocab in $VOCABS; do
    for model in $MODELS; do
      echo "=== ${model} | vocab=${vocab} | seed=${seed} ==="
      pids=(); gpu=0
      for tok in $TOK_TYPES; do
          launch "$gpu" "$model" "$tok" "$vocab" "$seed" & pids+=($!)
          gpu=$(( (gpu + 1) % CVL_NGPU ))
          (( ${#pids[@]} % CVL_NGPU == 0 )) && { for p in "${pids[@]}"; do wait "$p" || echo "[WARN] pid $p failed"; done; pids=(); }
      done
      for p in "${pids[@]}"; do wait "$p" || echo "[WARN] pid $p failed"; done
    done
  done
done
echo "=== training complete ==="
