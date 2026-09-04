#!/usr/bin/env python3
"""
Train BPE Tokenizers (All Variants) & Tokenize Data

Trains BPE tokenizers using all three scoring variants:
- count (bpe) - Standard BPE (most frequent pair)
- greedy_ll_exact - Rank by exact ΔLL
- greedy_ll_approx - Rank by approx ΔLL (PMI-like)

Usage:
    python train_bottomupll_tokenizers.py
    python train_bottomupll_tokenizers.py --vocab-sizes 8000 32000
    python train_bottomupll_tokenizers.py --methods bpe bottomupll-exact
    python train_bottomupll_tokenizers.py --skip-tokenization
"""

import os
import argparse
import gc
from pathlib import Path

from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm


# ===================
# DEFAULT CONFIGURATION
# ===================

DEFAULT_RAW_DATA_DIR = Path(os.environ.get("CVL_RAW_EN", "data/fineweb-edu-raw"))
DEFAULT_TOKENIZER_DIR = Path(os.environ.get("CVL_TOKENIZERS", "tokenizers"))
DEFAULT_DATA_DIR = Path(os.environ.get("CVL_DATA", "data"))

DEFAULT_VOCAB_SIZES = [8_000, 32_000, 128_000]

SCORING_METHODS = {
    "bpe": {"score_by": "count", "stop_by": "vocab_size"},
    "bottomupll-exact": {"score_by": "greedy_ll_exact", "stop_by": "vocab_size"},
    "bottomupll-approx": {"score_by": "greedy_ll_approx", "stop_by": "vocab_size"},
}

SPECIAL_TOKENS = [
    "<|endoftext|>",  # EOS token (id=0)
    "<|padding|>",    # PAD token (id=1)
]

DEFAULT_NUM_PROC = 8


# ===================
# FUNCTIONS
# ===================

def get_training_corpus(train_raw, batch_size=1000):
    """Yields batches of text for tokenizer training."""
    total_batches = (len(train_raw) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(train_raw), batch_size), total=total_batches, desc="Reading corpus", leave=False):
        yield train_raw[i:i+batch_size]["text"]


def train_bpe_tokenizer(vocab_size: int, corpus_iterator, score_by: str, stop_by: str) -> Tokenizer:
    """Train a byte-level BPE tokenizer with specified scoring method."""

    # Initialize byte-level BPE model
    tokenizer = Tokenizer(models.BPE())

    # Normalizer: NFC unicode normalization
    tokenizer.normalizer = NFC()

    # Pre-tokenizer: Byte-level (like GPT-2)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: Byte-level decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer (tracking disabled)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        score_by=score_by,
        stop_by=stop_by,
    )

    # Train
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # Post-processor: add EOS handling
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, vocab_size: int, output_dir: Path, name_prefix: str):
    """Save tokenizer in HuggingFace format."""

    name = f"{name_prefix}-{vocab_size // 1000}k"
    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)

    # Wrap in PreTrainedTokenizerFast for HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        bos_token="<|endoftext|>",
    )

    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")

    return save_path, hf_tokenizer


def tokenize_dataset(dataset, tokenizer, num_proc=8):
    """Tokenize a dataset, keeping uid."""

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        return {
            "input_ids": tokens["input_ids"],
            "uid": examples["uid"],
        }

    return dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizers with different scoring methods")
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR,
                        help="Path to raw data directory")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR,
                        help="Output directory for tokenizers")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Output directory for tokenized data")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES,
                        help="Vocab sizes to train (e.g., 8000 32000 128000)")
    parser.add_argument("--methods", type=str, nargs="+", default=list(SCORING_METHODS.keys()),
                        choices=list(SCORING_METHODS.keys()),
                        help="Scoring methods to train")
    parser.add_argument("--num-proc", type=int, default=DEFAULT_NUM_PROC,
                        help="Number of processes for tokenization")
    parser.add_argument("--skip-tokenization", action="store_true",
                        help="Skip dataset tokenization (only train tokenizers)")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip compression comparison")

    args = parser.parse_args()

    # ===================
    # LOAD DATA
    # ===================
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    train_raw = load_from_disk(str(args.raw_data_dir / "train"))
    val_raw = load_from_disk(str(args.raw_data_dir / "val"))
    test_raw = load_from_disk(str(args.raw_data_dir / "test"))

    print(f"Train: {len(train_raw):,} documents")
    print(f"Val:   {len(val_raw):,} documents")
    print(f"Test:  {len(test_raw):,} documents")

    # ===================
    # TRAIN TOKENIZERS
    # ===================
    print("\n" + "="*60)
    print("TRAINING TOKENIZERS")
    print("="*60)

    all_trained_tokenizers = {}
    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    total_combinations = len(args.methods) * len(args.vocab_sizes)

    with tqdm(total=total_combinations, desc="Training tokenizers") as pbar:
        for method_name in args.methods:
            method = SCORING_METHODS[method_name]
            score_by = method["score_by"]
            stop_by = method["stop_by"]

            print(f"\n{'#'*60}")
            print(f"# TRAINING: {method_name.upper()}")
            print(f"# score_by={score_by}, stop_by={stop_by}")
            print(f"{'#'*60}")

            for vocab_size in args.vocab_sizes:
                pbar.set_description(f"Training {method_name}-{vocab_size // 1000}k")
                print(f"\n{'='*50}")
                print(f"Training {method_name}-{vocab_size // 1000}k")
                print(f"{'='*50}")

                # Train
                tokenizer = train_bpe_tokenizer(
                    vocab_size,
                    get_training_corpus(train_raw),
                    score_by=score_by,
                    stop_by=stop_by,
                )

                # Save
                save_path, hf_tokenizer = save_tokenizer(tokenizer, vocab_size, args.tokenizer_dir, method_name)
                all_trained_tokenizers[(method_name, vocab_size)] = (save_path, hf_tokenizer)

                # Quick test
                test_text = "Hello, world! This is a test. 你好世界"
                tokens = hf_tokenizer.encode(test_text)
                print(f"Test: '{test_text}'")
                print(f"Tokens ({len(tokens)}): {tokens[:15]}...")
                print(f"Vocab size: {len(hf_tokenizer):,}")

                # Free memory from training
                del tokenizer
                gc.collect()

                pbar.update(1)

    # ===================
    # TOKENIZE DATASETS
    # ===================
    if not args.skip_tokenization:
        print("\n" + "="*60)
        print("TOKENIZING DATASETS")
        print("="*60)

        total_tokenizers = len(all_trained_tokenizers)

        with tqdm(total=total_tokenizers, desc="Tokenizing datasets") as pbar:
            for (method_name, vocab_size), (tok_path, tokenizer) in all_trained_tokenizers.items():
                pbar.set_description(f"Tokenizing {method_name}-{vocab_size // 1000}k")
                print(f"\n{'='*50}")
                print(f"Tokenizing with {method_name}-{vocab_size // 1000}k")
                print(f"{'='*50}")

                output_dir = args.data_dir / f"fineweb-edu-{method_name}-{vocab_size // 1000}k"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Tokenize each split
                print("Tokenizing train...")
                train_tok = tokenize_dataset(train_raw, tokenizer, args.num_proc)
                train_tok.save_to_disk(output_dir / "train")

                print("Tokenizing val...")
                val_tok = tokenize_dataset(val_raw, tokenizer, args.num_proc)
                val_tok.save_to_disk(output_dir / "val")

                print("Tokenizing test...")
                test_tok = tokenize_dataset(test_raw, tokenizer, args.num_proc)
                test_tok.save_to_disk(output_dir / "test")

                # Stats with progress bars
                print("Counting tokens...")
                train_tokens = sum(len(x) for x in tqdm(train_tok["input_ids"], desc="  Train tokens", leave=False))
                val_tokens = sum(len(x) for x in tqdm(val_tok["input_ids"], desc="  Val tokens", leave=False))
                test_tokens = sum(len(x) for x in tqdm(test_tok["input_ids"], desc="  Test tokens", leave=False))

                print(f"\nSaved to {output_dir}")
                print(f"  Train: {train_tokens:,} tokens ({train_tokens/1e9:.2f}B)")
                print(f"  Val:   {val_tokens:,} tokens ({val_tokens/1e6:.0f}M)")
                print(f"  Test:  {test_tokens:,} tokens ({test_tokens/1e6:.0f}M)")

                # Free memory from tokenized datasets
                del train_tok, val_tok, test_tok
                gc.collect()

                pbar.update(1)

        # Free memory from HF tokenizers after tokenization
        del all_trained_tokenizers
        gc.collect()
    else:
        # Free memory even if skipping tokenization
        del all_trained_tokenizers
        gc.collect()

    # ===================
    # COMPARISON
    # ===================
    if not args.skip_comparison and len(args.methods) > 1:
        print("\n" + "="*60)
        print("COMPRESSION COMPARISON")
        print("="*60)

        from transformers import AutoTokenizer

        sample_texts = train_raw[:1000]["text"]

        print("\nCompression comparison (total tokens on 1000 docs, lower is better):")
        print("="*70)

        for vocab_size in tqdm(args.vocab_sizes, desc="Comparing vocab sizes"):
            print(f"\nVocab size: {vocab_size // 1000}k")
            print("-" * 50)

            results = {}
            for method_name in args.methods:
                try:
                    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir / f"{method_name}-{vocab_size // 1000}k")
                    total_tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in tqdm(sample_texts, desc=f"  {method_name}", leave=False))
                    results[method_name] = total_tokens
                    del tok
                    gc.collect()
                except Exception as e:
                    print(f"  {method_name}: Error - {e}")

            # Find best (lowest token count)
            if results:
                best = min(results, key=results.get)
                baseline = results.get("bpe", list(results.values())[0])

                for name, tokens in results.items():
                    diff_pct = (tokens - baseline) / baseline * 100 if baseline else 0
                    marker = " <-- best" if name == best else ""
                    print(f"  {name:20s}: {tokens:,} tokens ({diff_pct:+.2f}%){marker}")

        # Free memory from comparison
        del sample_texts
        gc.collect()

    # ===================
    # SUMMARY
    # ===================
    print("\n" + "="*60)
    print("TOKENIZER TRAINING COMPLETE")
    print("="*60)
    print(f"\nTokenizers saved to: {args.tokenizer_dir.absolute()}")
    if not args.skip_tokenization:
        print(f"Tokenized data saved to: {args.data_dir.absolute()}")
    print("\nCreated:")

    for method_name in args.methods:
        method = SCORING_METHODS[method_name]
        print(f"\n  {method_name.upper()} (score_by={method['score_by']}):")
        for vocab_size in args.vocab_sizes:
            name = f"{method_name}-{vocab_size // 1000}k"
            print(f"    - {name}")


if __name__ == "__main__":
    main()
