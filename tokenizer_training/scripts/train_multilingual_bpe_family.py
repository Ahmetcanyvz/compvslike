#!/usr/bin/env python3
"""
Train BPE tokenizers (all 3 scoring variants) on a multilingual mix.

Corpus: 1B English (FineWeb-Edu) + 250M each from cmn, deu, spa, tur = ~2B total
Token counts estimated via GPT-2 tokenizer (same as the original FineWeb pipeline).

Methods:
  - bpe       (standard BPE, most frequent pair)
  - bottomupll-exact  (rank by exact ΔLL)
  - bottomupll-approx (rank by approx ΔLL / PMI-like)

Memory-efficient pattern:
  - The mixed corpus is built once and saved to a cache directory as an
    Arrow dataset (memory-mapped, not held in Python memory).
  - Each training run loads via load_from_disk() and iterates batches
    directly from the mmap'd file — matches the original
    train_bottomupll_tokenizers.py pattern.
  - The cache is reused on subsequent runs (delete it manually if you want
    to rebuild with different token budgets).

Output: <tokenizer-dir>/<method>-<suffix>-128k/
"""

import os
import argparse
import gc
from pathlib import Path

from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tqdm.auto import tqdm


# ===================
# CONFIGURATION
# ===================

DEFAULT_ENGLISH_DIR = Path(os.environ.get("CVL_RAW_EN", "data/fineweb-edu-raw")) / "train"
DEFAULT_MULTI_DIR = Path(os.environ.get("CVL_RAW_MULTI", "data/multilingual-raw"))
DEFAULT_TOKENIZER_DIR = Path(os.environ.get("CVL_TOKENIZERS", "tokenizers"))
DEFAULT_CORPUS_CACHE = Path(os.environ.get("CVL_DATA", "data")) / "multilingual-corpus-cache"

MULTI_LANGS = ["cmn", "deu", "spa", "tur"]

DEFAULT_ENGLISH_TOKENS = 1_000_000_000   # 1B
DEFAULT_PER_LANG_TOKENS = 250_000_000    # 250M per language (4 × 250M = 1B total)
DEFAULT_VOCAB_SIZE = 128_000

SCORING_METHODS = {
    "bpe":       {"score_by": "count",             "stop_by": "vocab_size"},
    "bottomupll-exact":  {"score_by": "greedy_ll_exact",   "stop_by": "vocab_size"},
    "bottomupll-approx": {"score_by": "greedy_ll_approx",  "stop_by": "vocab_size"},
}

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
]

ESTIMATOR_TOKENIZER = "gpt2"


# ===================
# CORPUS BUILDING (runs once, saves to disk)
# ===================

def collect_from_source(dataset, target_tokens, estimator, lang_label, next_uid,
                        min_doc_tokens=50):
    """Stream a source dataset, collect docs into a list of dicts until target hit.

    Returns (list of {"text", "uid", "lang"}, total_tokens).
    """
    records = []
    total_tokens = 0
    pbar = tqdm(total=target_tokens, unit="tok", desc=lang_label)

    for example in dataset:
        text = example.get("text") or ""
        if not text:
            continue
        est_tokens = len(estimator.encode(text, add_special_tokens=False))
        if est_tokens < min_doc_tokens:
            continue
        records.append({"text": text, "uid": next_uid + len(records), "lang": lang_label})
        total_tokens += est_tokens
        pbar.update(est_tokens)
        if total_tokens >= target_tokens:
            break
    pbar.close()
    return records, total_tokens


def build_and_save_corpus(english_dir, multi_dir, english_tokens, per_lang_tokens,
                          cache_dir):
    """Build the mixed corpus by streaming each source and save as an Arrow dataset."""
    print(f"Loading estimator tokenizer: {ESTIMATOR_TOKENIZER}")
    estimator = AutoTokenizer.from_pretrained(ESTIMATOR_TOKENIZER)

    all_records = []
    totals = {}

    # English
    print(f"\nStreaming English from {english_dir}")
    en_ds = load_from_disk(str(english_dir))
    print(f"  {len(en_ds):,} documents available")
    en_recs, en_tok = collect_from_source(en_ds, english_tokens, estimator,
                                          "eng", next_uid=len(all_records))
    all_records.extend(en_recs)
    totals["eng"] = en_tok
    print(f"  Collected {len(en_recs):,} docs ({en_tok:,} tokens)")
    del en_ds, en_recs
    gc.collect()

    # Multilingual
    for lang in MULTI_LANGS:
        lang_path = multi_dir / lang / "train"
        print(f"\nStreaming {lang} from {lang_path}")
        lang_ds = load_from_disk(str(lang_path))
        print(f"  {len(lang_ds):,} documents available")
        recs, tok = collect_from_source(lang_ds, per_lang_tokens, estimator,
                                        lang, next_uid=len(all_records))
        all_records.extend(recs)
        totals[lang] = tok
        print(f"  Collected {len(recs):,} docs ({tok:,} tokens)")
        del lang_ds, recs
        gc.collect()

    print(f"\nTotal: {len(all_records):,} documents, "
          f"{sum(totals.values()):,} estimated tokens")

    # Save as Arrow dataset, then free Python list
    print(f"\nSaving corpus to {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(all_records)
    del all_records
    gc.collect()
    dataset.save_to_disk(str(cache_dir))
    del dataset
    gc.collect()
    print("Corpus saved.")


def corpus_iterator_from_disk(cache_dir, batch_size=1000):
    """Yield batches of texts from the cached Arrow dataset (memory-mapped)."""
    ds = load_from_disk(str(cache_dir))
    total_batches = (len(ds) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(ds), batch_size), total=total_batches,
                  desc="Feeding corpus", leave=False):
        yield ds[i:i + batch_size]["text"]


# ===================
# TRAINING
# ===================

def train_bpe_tokenizer(vocab_size, cache_dir, score_by, stop_by):
    """Train a byte-level BPE tokenizer with the specified scoring method."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        score_by=score_by,
        stop_by=stop_by,
    )

    tokenizer.train_from_iterator(
        corpus_iterator_from_disk(cache_dir),
        trainer=trainer,
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def save_tokenizer(tokenizer, name, output_dir):
    """Save as HuggingFace PreTrainedTokenizerFast."""
    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        bos_token="<|endoftext|>",
    )
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
    return save_path


# ===================
# MAIN
# ===================

def main():
    parser = argparse.ArgumentParser(description="Train multilingual BPE tokenizers")
    parser.add_argument("--english-dir", type=Path, default=DEFAULT_ENGLISH_DIR)
    parser.add_argument("--multi-dir", type=Path, default=DEFAULT_MULTI_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--corpus-cache", type=Path, default=DEFAULT_CORPUS_CACHE,
                        help="Where to cache the prepared Arrow dataset (reused across runs)")
    parser.add_argument("--english-tokens", type=int, default=DEFAULT_ENGLISH_TOKENS)
    parser.add_argument("--per-lang-tokens", type=int, default=DEFAULT_PER_LANG_TOKENS)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--methods", type=str, nargs="+",
                        default=list(SCORING_METHODS.keys()),
                        choices=list(SCORING_METHODS.keys()))
    parser.add_argument("--name-suffix", type=str, default="multi",
                        help="Suffix in saved tokenizer name, e.g. '<method>-<suffix>-128k'")
    parser.add_argument("--rebuild-corpus", action="store_true",
                        help="Force rebuilding the cached corpus")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip methods whose output directory already exists")
    args = parser.parse_args()

    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build corpus (once, cached on disk) ----
    corpus_ready = args.corpus_cache.exists() and (args.corpus_cache / "state.json").exists()
    if corpus_ready and not args.rebuild_corpus:
        print(f"Using cached corpus at {args.corpus_cache}")
        ds = load_from_disk(str(args.corpus_cache))
        print(f"  {len(ds):,} documents loaded (memory-mapped)")
        del ds
    else:
        if args.rebuild_corpus and args.corpus_cache.exists():
            import shutil
            print(f"Removing existing cache at {args.corpus_cache}")
            shutil.rmtree(args.corpus_cache)
        print("=" * 60)
        print("BUILDING MIXED CORPUS")
        print("=" * 60)
        print(f"Target: {args.english_tokens:,} English + "
              f"{args.per_lang_tokens:,} × {len(MULTI_LANGS)} multilingual "
              f"= ~{args.english_tokens + args.per_lang_tokens * len(MULTI_LANGS):,} tokens")
        build_and_save_corpus(
            args.english_dir, args.multi_dir,
            args.english_tokens, args.per_lang_tokens,
            args.corpus_cache,
        )

    # ---- Train each method (streams from the cached Arrow dataset) ----
    print("\n" + "=" * 60)
    print("TRAINING TOKENIZERS")
    print("=" * 60)

    vocab_k = args.vocab_size // 1000
    for method_name in args.methods:
        name = f"{method_name}-{args.name_suffix}-{vocab_k}k"
        out_path = args.tokenizer_dir / name

        if args.skip_existing and out_path.exists():
            print(f"\n[SKIP] {name} already exists at {out_path}")
            continue

        cfg = SCORING_METHODS[method_name]
        print(f"\n{'#' * 60}")
        print(f"# TRAINING: {method_name.upper()}")
        print(f"# score_by={cfg['score_by']}, stop_by={cfg['stop_by']}")
        print(f"{'#' * 60}")

        tokenizer = train_bpe_tokenizer(
            args.vocab_size, args.corpus_cache,
            score_by=cfg["score_by"],
            stop_by=cfg["stop_by"],
        )

        save_tokenizer(tokenizer, name, args.tokenizer_dir)

        # Quick test
        test_text = "Hello, world! Hallo, Welt! ¡Hola, mundo! Merhaba dünya! 你好世界"
        output = tokenizer.encode(test_text)
        print(f"Test: {test_text!r}")
        print(f"  Tokens ({len(output.tokens)}): {output.tokens[:20]}...")

        del tokenizer
        gc.collect()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for method_name in args.methods:
        name = f"{method_name}-{args.name_suffix}-{vocab_k}k"
        print(f"  {args.tokenizer_dir / name}")


if __name__ == "__main__":
    main()
