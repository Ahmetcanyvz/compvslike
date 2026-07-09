#!/usr/bin/env bash
#SBATCH --job-name=exploration
#SBATCH --partition=normal
#SBATCH --account=a0229
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/exploration_%j.out
#SBATCH --error=logs/exploration_%j.err
#SBATCH --container-writable
#SBATCH --environment=lm_trainer_env

# Usage: sbatch run_sbatch_exploration.sh <VARIANT>
#   VARIANT = "english" or "multi"
# Example:
#   sbatch run_sbatch_exploration.sh english
#   sbatch run_sbatch_exploration.sh multi

set -euo pipefail

WORK_DIR="/iopsstor/scratch/cscs/ayavuz/compvslike"
VARIANT="${1:-english}"

cd "$WORK_DIR"
pip install -e . --no-deps
mkdir -p logs

OUT_DIR="exploration_results/${VARIANT}"
mkdir -p "$OUT_DIR"

export LM_TRAINER_ROOT="$WORK_DIR"
export EXPLORATION_VARIANT="$VARIANT"

cd scripts/exploration

# Multi data has no unified `test/` — only per-language test_eng, test_deu, test_spa, test_tur, test_cmn.
# Merge them into test_merged/ once per tokenizer (skipped if already exists).
if [[ "$VARIANT" == "multi" ]]; then
    SPLIT_ARG="--split test_merged"
    echo "=== Merging per-language test splits into test_merged/ for each multi tokenizer ==="
    python - <<'PYEOF'
from datasets import load_from_disk, concatenate_datasets
from pathlib import Path
import os

ROOT = Path(os.environ["LM_TRAINER_ROOT"])
LANGS = ["eng", "deu", "spa", "tur", "cmn"]
TOKS = [
    "bpe_count-multi-128k",
    "compmax-multi-128k",
    "greedyll-exact-multi-128k",
    "unigramlm-multi-128k",
]

for tok in TOKS:
    base = ROOT / "data" / "multilingual" / f"multilingual-{tok}"
    out = base / "test_merged"
    if out.exists():
        print(f"  [SKIP] {out} already exists")
        continue
    parts = []
    for lang in LANGS:
        p = base / f"test_{lang}"
        if not p.exists():
            print(f"  [WARN] missing {p}")
            continue
        parts.append(load_from_disk(str(p)))
    if not parts:
        print(f"  [SKIP] no parts for {tok}")
        continue
    merged = concatenate_datasets(parts)
    merged.save_to_disk(str(out))
    print(f"  [OK] merged {len(parts)} languages -> {out} ({len(merged):,} docs)")
PYEOF
else
    SPLIT_ARG=""
fi

echo "=== Exploration variant: ${VARIANT} ==="
echo "=== LM_TRAINER_ROOT: ${LM_TRAINER_ROOT} ==="
echo "=== Split arg: ${SPLIT_ARG:-(default test)} ==="
echo ""

run_step() {
    local name=$1
    shift
    local logfile="${WORK_DIR}/${OUT_DIR}/${name}.log"
    echo "[RUN] ${name} -> ${logfile}"
    if python "$@" > "$logfile" 2>&1; then
        echo "[OK]  ${name}"
    else
        echo "[FAIL] ${name}  (see ${logfile})"
    fi
}

# Scripts that take --split (i.e., read pre-tokenized data)
run_step compression_stats     compression_stats.py $SPLIT_ARG
run_step zipf_and_entropy      zipf_and_entropy.py $SPLIT_ARG
# Scripts that don't read pre-tokenized data (just tokenizer.json)
run_step token_length_dist     token_length_dist.py
run_step vocab_differences     vocab_differences.py
run_step merge_overlap         merge_overlap.py
run_step kill_ratio            kill_ratio.py
run_step tokenization_examples tokenization_examples.py

echo ""
echo "=== Done. Results in ${OUT_DIR}/ ==="
