#!/usr/bin/env bash
set -euo pipefail

cd /iopsstor/scratch/cscs/ayavuz/compvslike
pip install -e . --no-deps

echo "=== Preparing multilingual tokenized data ==="
python scripts/prepare_multilingual.py \
    -t tokenizers/bpe_count-multi-128k \
    -t tokenizers/compmax-multi-128k \
    -t tokenizers/greedyll-exact-multi-128k \
    -t tokenizers/unigramlm-multi-128k \
    -o data/multilingual \
    --raw-data-dir data/multilingual-raw \
    --total-tokens 20000000000 \
    --eng-raw-dir data/fineweb-edu-raw
echo "=== Done ==="
