"""Shared configuration for all exploration scripts.

Two environment variables control behavior:
  LM_TRAINER_ROOT     — root of the lm-trainer checkout (default: parent of this dir's parent)
  EXPLORATION_VARIANT — "english" (default) or "multi"

Usage:
    # English (default)
    python compression_stats.py

    # Multilingual
    EXPLORATION_VARIANT=multi python compression_stats.py

    # On clariden
    LM_TRAINER_ROOT=/iopsstor/scratch/cscs/ayavuz/compvslike EXPLORATION_VARIANT=multi python compression_stats.py
"""

import json
import os
from pathlib import Path

# Root of the repo. Defaults to two levels up from this file
# (lm-trainer/scripts/exploration/config.py -> lm-trainer).
ROOT = Path(os.environ.get("LM_TRAINER_ROOT", Path(__file__).resolve().parent.parent.parent))
VARIANT = os.environ.get("EXPLORATION_VARIANT", "english")


def _english_paths():
    tok_root = ROOT / "tokenizers"
    data_root = ROOT / "data"
    tokenizers = {
        "bpe-128k": tok_root / "bpe-128k",
        "greedyll-exact-128k": tok_root / "greedyll-exact-128k",
        "greedyll-approx-128k": tok_root / "greedyll-approx-128k",
        "unigramlm-128k": tok_root / "unigramlm-128k",
        "compmax-128k": tok_root / "compmax-128k",
    }
    data = {
        "bpe-128k": data_root / "fineweb-edu-bpe-128k",
        "greedyll-exact-128k": data_root / "fineweb-edu-greedyll-exact-128k",
        "greedyll-approx-128k": data_root / "fineweb-edu-greedyll-approx-128k",
        "unigramlm-128k": data_root / "fineweb-edu-unigramlm-128k",
        "compmax-128k": data_root / "fineweb-edu-compmax-128k",
    }
    short = {
        "bpe-128k": "BPE",
        "greedyll-exact-128k": "Exact",
        "greedyll-approx-128k": "Approx",
        "unigramlm-128k": "Unigram",
        "compmax-128k": "CompMax",
    }
    bpe_methods = ["bpe-128k", "greedyll-exact-128k", "greedyll-approx-128k"]
    raw_test = data_root / "fineweb-edu-raw" / "test"
    return tokenizers, data, short, bpe_methods, raw_test


def _multi_paths():
    tok_root = ROOT / "tokenizers"
    data_root = ROOT / "data" / "multilingual"
    tokenizers = {
        "bpe_count-multi-128k": tok_root / "bpe_count-multi-128k",
        "greedyll-exact-multi-128k": tok_root / "greedyll-exact-multi-128k",
        "unigramlm-multi-128k": tok_root / "unigramlm-multi-128k",
        "compmax-multi-128k": tok_root / "compmax-multi-128k",
    }
    data = {
        "bpe_count-multi-128k": data_root / "multilingual-bpe_count-multi-128k",
        "greedyll-exact-multi-128k": data_root / "multilingual-greedyll-exact-multi-128k",
        "unigramlm-multi-128k": data_root / "multilingual-unigramlm-multi-128k",
        "compmax-multi-128k": data_root / "multilingual-compmax-multi-128k",
    }
    short = {
        "bpe_count-multi-128k": "BPE",
        "greedyll-exact-multi-128k": "Exact",
        "unigramlm-multi-128k": "Unigram",
        "compmax-multi-128k": "CompMax",
    }
    # Multilingual setup currently has no greedyll-approx variant
    bpe_methods = ["bpe_count-multi-128k", "greedyll-exact-multi-128k"]
    raw_test = ROOT / "data" / "fineweb-edu-raw" / "test"
    return tokenizers, data, short, bpe_methods, raw_test


if VARIANT == "english":
    TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS, RAW_TEST_PATH = _english_paths()
elif VARIANT == "multi":
    TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS, RAW_TEST_PATH = _multi_paths()
else:
    raise ValueError(f"Unknown EXPLORATION_VARIANT={VARIANT!r}; expected 'english' or 'multi'")

# Cast to str for libraries that expect string paths.
TOKENIZER_PATHS = {k: str(v) for k, v in TOKENIZER_PATHS.items()}
DATA_PATHS = {k: str(v) for k, v in DATA_PATHS.items()}
RAW_TEST_PATH = str(RAW_TEST_PATH)

ALL_METHODS = list(TOKENIZER_PATHS.keys())


def load_vocab(tokenizer_path):
    """Load vocab as {token_str: token_id} from tokenizer.json."""
    with open(f"{tokenizer_path}/tokenizer.json") as f:
        tj = json.load(f)
    if tj["model"]["type"] == "BPE":
        return tj["model"]["vocab"]
    else:  # Unigram
        return {piece[0]: i for i, piece in enumerate(tj["model"]["vocab"])}
