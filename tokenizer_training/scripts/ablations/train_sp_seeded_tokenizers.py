#!/usr/bin/env python3
"""
Train CompMax + UnigramLM tokenizers initialized with SentencePiece's seed.

Pipeline:
  1. Pre-tokenize the corpus with HF ByteLevel (so SP sees Ġ-style words).
  2. Train SentencePiece briefly to extract its seed vocabulary
     (esaxx-based, but per-word due to SP's pre-splitting).
  3. Pass that seed_vocab to both CompressionTrainer and UnigramTrainer.

Same settings as train_compression_tokenizers.py except for the seed source.

Output: <tokenizer-dir>/<method>_sentencepiece-{N}k/
   where method is topdowncomp or unigramlm
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm
from datasets import load_from_disk
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors, AddedToken
from tokenizers.models import Unigram
from tokenizers.trainers import CompressionTrainer, UnigramTrainer
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm


# ===================
# DEFAULT CONFIGURATION
# ===================

DEFAULT_RAW_DATA_DIR = Path(os.environ.get("CVL_RAW_EN", "data/fineweb-edu-raw"))
DEFAULT_TOKENIZER_DIR = Path(os.environ.get("CVL_TOKENIZERS", "tokenizers"))
DEFAULT_SEED_CACHE_DIR = Path(os.environ.get("CVL_SP_SEEDS", "sp_seeds"))

DEFAULT_VOCAB_SIZES = [8_000, 32_000, 128_000]
# Request 1M from SP — it will plateau around ~150-200k for English.
# We just use whatever SP gives us.
DEFAULT_SP_SEED_REQUEST = 1_000_000

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
]

MAX_PIECE_LENGTH = 16


# ===================
# CORPUS / SEED EXTRACTION
# ===================

def get_training_corpus(train_raw, batch_size=1000):
    """Yields batches of text for tokenizer training."""
    total_batches = (len(train_raw) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(train_raw), batch_size), total=total_batches,
                  desc="Reading corpus", leave=False):
        yield train_raw[i:i + batch_size]["text"]


def write_pretokenized_corpus(train_raw, out_path: Path, batch_size: int = 5000):
    """
    Apply NFC + HF ByteLevel pre-tokenization to the corpus and write each
    pre-token on its own line. SentencePiece will treat each line as a sentence.
    """
    from tokenizers.normalizers import NFC as NFCNorm
    nfc = NFCNorm()
    pre_tok = ByteLevelPre(add_prefix_space=False)

    print(f"Writing pre-tokenized corpus to {out_path}")
    n_pretokens = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for batch in get_training_corpus(train_raw, batch_size=batch_size):
            for text in batch:
                # Apply NFC normalizer
                normalized = nfc.normalize_str(text)
                for word, _ in pre_tok.pre_tokenize_str(normalized):
                    f.write(word + "\n")
                    n_pretokens += 1
    print(f"  Wrote {n_pretokens:,} pre-tokens")
    return n_pretokens


def extract_sp_seed(pretok_path: Path, seed_size: int,
                    cache_dir: Path) -> list[str]:
    """
    Train SentencePiece on the pre-tokenized corpus and return its vocab as
    a seed list. Cache to disk so it's reusable across runs/methods.
    """
    cache_file = cache_dir / f"sp_seed_{seed_size}.txt"
    if cache_file.exists():
        print(f"Loading cached SP seed from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            seed = [line.rstrip("\n") for line in f if line.rstrip("\n")]
        print(f"  Loaded {len(seed):,} seed tokens")
        return seed

    print(f"Training SentencePiece to extract seed (target {seed_size:,})...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = os.path.join(tmpdir, "sp_seed")
        spm.SentencePieceTrainer.train(
            input=str(pretok_path),
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=seed_size,
            character_coverage=1.0,
            # Cap SP input to avoid OOM on huge corpora. SP samples this many
            # pre-tokens uniformly at random — plenty for seed extraction.
            input_sentence_size=100_000_000,
            shuffle_input_sentence=True,
            max_sentence_length=100000,
            max_sentencepiece_length=MAX_PIECE_LENGTH,
            num_threads=8,
            seed_sentencepiece_size=seed_size,
            shrinking_factor=0.95,
            num_sub_iterations=1,
            normalization_rule_name="identity",
            split_by_whitespace=False,
            split_by_unicode_script=False,
            split_by_number=False,
            split_digits=False,
            remove_extra_whitespaces=False,
            allow_whitespace_only_pieces=True,
            add_dummy_prefix=False,
            byte_fallback=False,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            train_extremely_large_corpus=True,
            hard_vocab_limit=False,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(model_prefix + ".model")

        seed = []
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            if piece in ("<unk>", "<s>", "</s>"):
                continue
            seed.append(piece)

    # Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        for tok in seed:
            f.write(tok + "\n")
    print(f"  Extracted {len(seed):,} seed tokens, cached at {cache_file}")
    return seed


# ===================
# TRAINING (SP-seeded)
# ===================

def train_topdowncomp_sp(vocab_size: int, corpus_iterator, seed_vocab: list[str]) -> Tokenizer:
    """CompressionTrainer initialized with SP seed vocab."""
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special = [AddedToken(s) for s in SPECIAL_TOKENS]

    trainer = CompressionTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        max_piece_length=MAX_PIECE_LENGTH,
        seed_vocab=seed_vocab,
        prune_ratio=0.1,
        min_prune=1,
        batch_recompute=True,
        initial_alphabet=ByteLevelPre.alphabet(),
        special_tokens=special,
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def train_unigramlm_sp(vocab_size: int, corpus_iterator, seed_vocab: list[str]) -> Tokenizer:
    """UnigramTrainer (EM) initialized with SP seed vocab."""
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special = [AddedToken(s) for s in SPECIAL_TOKENS]

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        max_piece_length=MAX_PIECE_LENGTH,
        initial_alphabet=ByteLevelPre.alphabet(),
        special_tokens=special,
        shrinking_factor=0.9,
        seed_vocab=seed_vocab,
    )

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


TRAIN_FNS = {
    "topdowncomp": train_topdowncomp_sp,
    "unigramlm": train_unigramlm_sp,
}


def save_tokenizer(tokenizer: Tokenizer, vocab_size: int, output_dir: Path, name_prefix: str):
    """Save as HuggingFace PreTrainedTokenizerFast (with the tokenizer_class fix)."""
    name = f"{name_prefix}_sentencepiece-{vocab_size // 1000}k"
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
    return save_path


# ===================
# MAIN
# ===================

def main():
    parser = argparse.ArgumentParser(description="Train SP-seeded topdowncomp/unigramlm tokenizers")
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--seed-cache-dir", type=Path, default=DEFAULT_SEED_CACHE_DIR)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--sp-seed-size", type=int, default=DEFAULT_SP_SEED_REQUEST,
                        help="Size to request from SP (it will plateau at ~150-200k for English)")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=list(TRAIN_FNS.keys()),
                        choices=list(TRAIN_FNS.keys()))
    parser.add_argument("--rebuild-seed", action="store_true",
                        help="Force re-extracting the SP seed even if cached")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tokenizers whose output already exists")
    args = parser.parse_args()

    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    args.seed_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load corpus ----
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_raw = load_from_disk(str(args.raw_data_dir / "train"))
    print(f"Train: {len(train_raw):,} documents")

    # ---- Pre-tokenize and write to disk for SP ----
    pretok_path = args.seed_cache_dir / "pretokenized.txt"
    if not pretok_path.exists() or args.rebuild_seed:
        print("\n" + "=" * 60)
        print("PRE-TOKENIZING CORPUS FOR SENTENCEPIECE")
        print("=" * 60)
        write_pretokenized_corpus(train_raw, pretok_path)
    else:
        print(f"\nUsing existing pre-tokenized corpus at {pretok_path}")

    # ---- Extract SP seed ----
    sp_seed_size = args.sp_seed_size
    print("\n" + "=" * 60)
    print(f"EXTRACTING SP SEED (requesting {sp_seed_size:,})")
    print("=" * 60)
    if args.rebuild_seed:
        cache_file = args.seed_cache_dir / f"sp_seed_{sp_seed_size}.txt"
        if cache_file.exists():
            cache_file.unlink()
    seed_vocab = extract_sp_seed(pretok_path, sp_seed_size, args.seed_cache_dir)
    print(f"\nSP returned {len(seed_vocab):,} seed tokens")

    # ---- Merge in ByteLevel alphabet (all 256 byte chars) ----
    # Required because seed_vocab bypasses the trainer's initial_alphabet injection.
    # Without this, rare bytes that SP never saw can't be encoded.
    bl_alphabet = ByteLevelPre.alphabet()
    seed_set = set(seed_vocab)
    added = 0
    for c in bl_alphabet:
        if c not in seed_set:
            seed_vocab.append(c)
            seed_set.add(c)
            added += 1
    print(f"Added {added} missing ByteLevel chars; final seed size: {len(seed_vocab):,}")

    # ---- Train each method × vocab_size ----
    print("\n" + "=" * 60)
    print("TRAINING SP-SEEDED TOKENIZERS")
    print("=" * 60)

    total = len(args.methods) * len(args.vocab_sizes)
    with tqdm(total=total, desc="Training") as pbar:
        for method_name in args.methods:
            train_fn = TRAIN_FNS[method_name]
            for vocab_size in args.vocab_sizes:
                name = f"{method_name}_sentencepiece-{vocab_size // 1000}k"
                out_path = args.tokenizer_dir / name

                if args.skip_existing and out_path.exists():
                    print(f"\n[SKIP] {name} already exists")
                    pbar.update(1)
                    continue

                if len(seed_vocab) < vocab_size:
                    print(f"\n[SKIP] SP seed has only {len(seed_vocab):,} tokens "
                          f"< vocab_size {vocab_size:,}. Skipping this size.")
                    pbar.update(1)
                    continue

                print(f"\n{'='*50}")
                print(f"Training {name}")
                print(f"{'='*50}")

                tokenizer = train_fn(
                    vocab_size,
                    get_training_corpus(train_raw),
                    seed_vocab=seed_vocab,
                )

                save_tokenizer(tokenizer, vocab_size, args.tokenizer_dir, method_name)

                test_text = "Hello, world! This is a test. 你好世界"
                hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
                tokens = hf.encode(test_text)
                print(f"Test: {test_text!r}")
                print(f"  Tokens ({len(tokens)}): {tokens[:15]}...")

                del tokenizer, hf
                gc.collect()
                pbar.update(1)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    for method_name in args.methods:
        for vocab_size in args.vocab_sizes:
            print(f"  {args.tokenizer_dir / f'{method_name}_sentencepiece-{vocab_size // 1000}k'}")


if __name__ == "__main__":
    main()
