#!/usr/bin/env python3
"""
Train CompMax with EXACT (context-aware) removal scoring at real scale, then
compare against the local d[t] approximation (existing topdowncomp-128k).

Exact scoring re-segments the actual corpus spans that use each token instead of
decomposing the token's string in isolation. We realize this with rand_scoring=True
and a large rand_sample_size (spans processed = min(sample_size, #spans), so a big
cap = all spans = exact per-word cost).

Settings match train_compression_tokenizers.py:
  NFC + ByteLevel(add_prefix_space=False), 1M seed, prune_ratio=0.1.

Output: <tokenizer-dir>/topdowncomp_exact-{N}k/
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

DEFAULT_VOCAB_SIZES = [128_000]
DEFAULT_SAMPLE_SIZE = 5_000   # spans re-segmented per token; larger = closer to exact

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>"]

BASE_TOKENS = 2_000_000_000
BASE_SEED = 42
BASE_TRAIN_RATIO = 0.95
BASE_VAL_RATIO = 0.025
BASE_MIN_TOKENS = 50


def download_base_data(output_dir: Path):
    if (output_dir / "train").exists() and (output_dir / "test").exists():
        print(f"Raw data already exists at {output_dir}")
        return
    print(f"Downloading base 2B tokens from FineWeb-Edu to {output_dir}...")
    estimator = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    documents, total_tokens = [], 0
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
    random.seed(BASE_SEED)
    random.shuffle(documents)
    n = len(documents)
    train_end = int(BASE_TRAIN_RATIO * n)
    val_end = int((BASE_TRAIN_RATIO + BASE_VAL_RATIO) * n)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in [("train", documents[:train_end]),
                       ("val", documents[train_end:val_end]),
                       ("test", documents[val_end:])]:
        Dataset.from_list(docs).save_to_disk(str(output_dir / name))
        print(f"  {name}: {len(docs):,} docs")


def get_training_corpus(train_raw, batch_size=1000):
    total = (len(train_raw) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(train_raw), batch_size), total=total,
                  desc="Reading corpus", leave=False):
        yield train_raw[i:i + batch_size]["text"]


def train_topdowncomp_exact(vocab_size, corpus_iterator, sample_size):
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
        prune_ratio=0.1,
        min_prune=1,
        batch_recompute=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=special,
        rand_scoring=True,          # empirical context-aware scoring
        rand_sample_size=sample_size,  # large => approaches exact
    )
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def save_tokenizer(tokenizer, output_dir: Path, name: str):
    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>", pad_token="<|padding|>", bos_token="<|endoftext|>",
    )
    hf.save_pretrained(save_path)
    cfg_path = save_path / "tokenizer_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Saved to {save_path}")


def compare(tokenizer_dir: Path, vocab_size: int, raw_dir: Path):
    vk = vocab_size // 1000
    local_path = tokenizer_dir / f"topdowncomp-{vk}k"
    exact_path = tokenizer_dir / f"topdowncomp_exact-{vk}k"
    if not local_path.exists():
        print(f"\n[compare] {local_path} not found — skip comparison.")
        return
    print("\n" + "=" * 60)
    print(f"COMPARE local (topdowncomp-{vk}k) vs exact (topdowncomp_exact-{vk}k)")
    print("=" * 60)
    local = PreTrainedTokenizerFast.from_pretrained(str(local_path))
    exact = PreTrainedTokenizerFast.from_pretrained(str(exact_path))
    vl, ve = set(local.get_vocab()), set(exact.get_vocab())
    common = vl & ve
    print(f"local vocab: {len(vl):,}")
    print(f"exact vocab: {len(ve):,}")
    print(f"common:      {len(common):,} ({len(common)/max(len(vl),len(ve))*100:.2f}%)")

    val = load_from_disk(str(raw_dir / "val"))
    sample = val[:5000]["text"]
    nl = sum(len(local.encode(t, add_special_tokens=False)) for t in tqdm(sample, desc="local", leave=False))
    ne = sum(len(exact.encode(t, add_special_tokens=False)) for t in tqdm(sample, desc="exact", leave=False))
    print(f"\nCompression on 5k val docs:")
    print(f"  local (d[t] approx): {nl:,}")
    print(f"  exact (context):     {ne:,}")
    d = nl - ne
    print(f"  exact saves: {d:,} ({d/nl*100:+.4f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Spans re-segmented per token (larger = closer to exact)")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    print(f"CompMax EXACT scoring (rand_sample_size={args.sample_size:,}), prune_ratio=0.1")
    print("=" * 60)

    download_base_data(args.raw_data_dir)
    train_raw = load_from_disk(str(args.raw_data_dir / "train"))
    print(f"Train: {len(train_raw):,} documents")

    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for vocab_size in args.vocab_sizes:
        name = f"topdowncomp_exact-{vocab_size // 1000}k"
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        tokenizer = train_topdowncomp_exact(
            vocab_size, get_training_corpus(train_raw), args.sample_size,
        )
        save_tokenizer(tokenizer, args.tokenizer_dir, name)

        test_text = "Hello, world! This is a test. 你好世界"
        hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        toks = hf.encode(test_text)
        print(f"Test: {test_text!r}\n  Tokens ({len(toks)}): {toks[:15]}...")
        del tokenizer, hf
        gc.collect()

        if not args.skip_compare:
            compare(args.tokenizer_dir, vocab_size, args.raw_data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
