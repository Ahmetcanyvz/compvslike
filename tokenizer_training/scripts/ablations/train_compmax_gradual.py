#!/usr/bin/env python3
"""
Train CompMax tokenizers with a small prune_ratio for gradual pruning.
Same settings as train_compression_tokenizers.py — only prune_ratio differs.

Names tokenizers as compmax_gradual_{pr}-{N}k where pr encodes prune_ratio,
e.g. prune_ratio=0.01 -> compmax_gradual_01-8k
     prune_ratio=0.001 -> compmax_gradual_001-8k
"""

import os
import argparse
import gc
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors, AddedToken
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer
from tokenizers.normalizers import NFC
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm


DEFAULT_RAW_DATA_DIR = Path(os.environ.get("CVL_RAW_EN", "data/fineweb-edu-raw"))
DEFAULT_TOKENIZER_DIR = Path(os.environ.get("CVL_TOKENIZERS", "tokenizers"))

DEFAULT_VOCAB_SIZES = [8_000, 32_000, 128_000]
DEFAULT_PRUNE_RATIO = 0.01   # gradual pruning: 1% per pass (default is 10%)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
]

# Dataset prep constants
BASE_TOKENS = 2_000_000_000
BASE_SEED = 42
BASE_TRAIN_RATIO = 0.95
BASE_VAL_RATIO = 0.025
BASE_MIN_TOKENS = 50


def download_base_data(output_dir: Path):
    """Stream 2B tokens from FineWeb-Edu, shuffle (seed=42), split 95/2.5/2.5%."""
    if (output_dir / "train").exists() and (output_dir / "test").exists():
        print(f"Raw data already exists at {output_dir}")
        return

    print(f"Downloading base 2B tokens from FineWeb-Edu to {output_dir}...")
    estimator = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    documents = []
    total_tokens = 0
    pbar = tqdm(total=BASE_TOKENS, unit="tok", desc="Downloading")
    for example in dataset:
        text = example["text"]
        est = len(estimator.encode(text, add_special_tokens=False))
        if est < BASE_MIN_TOKENS:
            continue
        documents.append({"text": text, "uid": len(documents)})
        total_tokens += est
        pbar.update(est)
        if total_tokens >= BASE_TOKENS:
            break
    pbar.close()

    print(f"Collected {len(documents):,} docs ({total_tokens/1e9:.2f}B tokens)")
    random.seed(BASE_SEED)
    random.shuffle(documents)
    n = len(documents)
    train_end = int(BASE_TRAIN_RATIO * n)
    val_end = int((BASE_TRAIN_RATIO + BASE_VAL_RATIO) * n)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in [
        ("train", documents[:train_end]),
        ("val", documents[train_end:val_end]),
        ("test", documents[val_end:]),
    ]:
        Dataset.from_list(docs).save_to_disk(str(output_dir / name))
        print(f"  {name}: {len(docs):,} docs")


def get_training_corpus(train_raw, batch_size=1000):
    total = (len(train_raw) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(train_raw), batch_size), total=total,
                  desc="Reading corpus", leave=False):
        yield train_raw[i:i + batch_size]["text"]


def train_compmax_gradual(vocab_size: int, corpus_iterator, prune_ratio: float) -> Tokenizer:
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special = [AddedToken(s) for s in SPECIAL_TOKENS]

    trainer = CompressionTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        max_piece_length=16,
        seed_size=1_000_000,
        prune_ratio=prune_ratio,
        min_prune=1,
        batch_recompute=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special,
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def save_tokenizer(tokenizer, vocab_size: int, output_dir: Path, name: str):
    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        bos_token="<|endoftext|>",
    )
    hf_tokenizer.save_pretrained(save_path)

    config_path = save_path / "tokenizer_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--prune-ratio", type=float, default=DEFAULT_PRUNE_RATIO,
                        help="Fraction to prune per pass (e.g. 0.01 = 1%%)")
    args = parser.parse_args()

    pr = args.prune_ratio
    # 0.01 -> "01", 0.001 -> "001", 0.1 -> "1"
    pr_tag = str(pr).split(".", 1)[1] if "." in str(pr) else str(pr)

    print(f"Training CompMax with prune_ratio={pr}")
    print("=" * 60)

    download_base_data(args.raw_data_dir)
    train_raw = load_from_disk(str(args.raw_data_dir / "train"))
    print(f"Train: {len(train_raw):,} documents")

    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for vocab_size in args.vocab_sizes:
        name = f"compmax_gradual_{pr_tag}-{vocab_size // 1000}k"
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")

        tokenizer = train_compmax_gradual(
            vocab_size,
            get_training_corpus(train_raw),
            prune_ratio=pr,
        )

        save_tokenizer(tokenizer, vocab_size, args.tokenizer_dir, name)

        test_text = "Hello, world! This is a test. 你好世界"
        hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tokens = hf.encode(test_text)
        print(f"Test: {test_text!r}")
        print(f"  Tokens ({len(tokens)}): {tokens[:15]}...")

        del tokenizer, hf
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
