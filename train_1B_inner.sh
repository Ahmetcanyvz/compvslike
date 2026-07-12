#!/usr/bin/env bash
# Runs INSIDE the container (via srun --environment). Does the actual training.
# Args: SEED TOK_NAME CONFIG_FILE PORT [RESUME_FLAG...]
set -e

SEED=$1
TOK_NAME=$2
CONFIG_FILE=$3
PORT=$4
shift 4
RESUME_FLAG="$*"

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
DATA_BASE="${WORK_DIR}/data"

cd "$WORK_DIR"
pip install -e . --no-deps

# Pre-create dataset metadata single-process (shuffle_seed = training seed)
# so the 4 DDP ranks load it instead of racing to create it.
echo "=== [seed ${SEED} / ${TOK_NAME}] Pre-creating dataset metadata ==="
python -c "
from src.data import PackedTokenDataset
for split, sd in [('train', ${SEED}), ('val', None)]:
    path = '${DATA_BASE}/fineweb-edu-${TOK_NAME}/' + split
    ds = PackedTokenDataset(path, seq_len=2048, eos_token_id=0, shuffle_seed=sd)
    print(f'  {split}: {len(ds)} sequences')
"

echo "=== Training: me1B-tied / ${TOK_NAME} / seed${SEED} ${RESUME_FLAG} ==="
export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_TIMEOUT=3600
export MASTER_ADDR=localhost
export MASTER_PORT="${PORT}"

torchrun --nproc_per_node=4 --master_addr=localhost --master_port="${PORT}" \
    -m src.train train "$CONFIG_FILE" --seed "$SEED" $RESUME_FLAG
