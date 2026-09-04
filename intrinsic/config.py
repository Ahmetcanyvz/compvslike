"""Shared configuration for the intrinsic analysis scripts.

These measure properties of the tokenisers themselves (compression, token
length, Zipf/entropy, vocabulary overlap), independently of any trained model.

Environment:
  CVL_TOKENIZERS      — directory holding the tokenisers (default: <repo>/tokenizers)
  CVL_DATA            — directory holding the tokenised corpora (default: <repo>/data)
  CVL_RAW_EN          — raw English corpus (default: <repo>/data/fineweb-edu-raw)
  CVL_VARIANT         — "english" (default) or "multi"

Usage:
    python intrinsic/compression_stats.py
    CVL_VARIANT=multi python intrinsic/compression_stats.py
"""

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TOKENIZER_ROOT = Path(os.environ.get("CVL_TOKENIZERS", REPO / "tokenizers"))
DATA_ROOT = Path(os.environ.get("CVL_DATA", REPO / "data"))
RAW_EN = Path(os.environ.get("CVL_RAW_EN", DATA_ROOT / "fineweb-edu-raw"))
VARIANT = os.environ.get("CVL_VARIANT", "english")

# Display names as they appear in the paper.
SHORT_NAMES = {
    "bpe": "BPE",
    "bottomupll-exact": "BottomUpLL",
    "bottomupll-approx": "BottomUpLL~",
    "topdowncomp": "TopDownComp",
    "unigramlm": "UnigramLM",
}

# Bottom-up (merging) methods; the merge-order analyses apply only to these.
BOTTOM_UP = ["bpe", "bottomupll-exact", "bottomupll-approx"]


def _english_paths():
    methods = ["bpe", "bottomupll-exact", "bottomupll-approx", "unigramlm", "topdowncomp"]
    sizes = ["8k", "32k", "128k"]
    tokenizers, data, short = {}, {}, {}
    for m in methods:
        for s in sizes:
            name = f"{m}-{s}"
            tokenizers[name] = TOKENIZER_ROOT / name
            data[name] = DATA_ROOT / f"fineweb-edu-{name}"
            short[name] = f"{SHORT_NAMES[m]}-{s}"
    # At 128k the size suffix is dropped, matching the paper's tables.
    for m in methods:
        short[f"{m}-128k"] = SHORT_NAMES[m]
    bpe_methods = [f"{m}-{s}" for m in BOTTOM_UP for s in sizes]
    return tokenizers, data, short, bpe_methods, RAW_EN / "test"


def _multi_paths():
    methods = ["bpe", "bottomupll-exact", "bottomupll-approx", "unigramlm", "topdowncomp"]
    data_root = DATA_ROOT / "multilingual"
    tokenizers, data, short = {}, {}, {}
    for m in methods:
        name = f"{m}-multi-128k"
        tokenizers[name] = TOKENIZER_ROOT / name
        data[name] = data_root / f"multilingual-{name}"
        short[name] = SHORT_NAMES[m]
    bpe_methods = [f"{m}-multi-128k" for m in BOTTOM_UP]
    return tokenizers, data, short, bpe_methods, RAW_EN / "test"


if VARIANT == "english":
    TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS, RAW_TEST_PATH = _english_paths()
elif VARIANT == "multi":
    TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS, RAW_TEST_PATH = _multi_paths()
else:
    raise ValueError(f"Unknown CVL_VARIANT={VARIANT!r}; expected 'english' or 'multi'")

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
    return {piece[0]: i for i, piece in enumerate(tj["model"]["vocab"])}
