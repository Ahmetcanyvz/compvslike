#!/usr/bin/env python3
"""Compare compression (token counts) across tokenizer methods on pre-tokenized data."""

import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
from config import TOKENIZER_PATHS, DATA_PATHS, SHORT, ALL_METHODS, BPE_METHODS, RAW_TEST_PATH


def main():
    parser = argparse.ArgumentParser(description="Compare compression across tokenizer methods")
    parser.add_argument("--tokenizers", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Tokenizer names to compare")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--baseline", type=str, default=BPE_METHODS[0])
    parser.add_argument("--sample-texts", type=int, default=0,
                        help="If > 0, also tokenize N raw texts for comparison")
    args = parser.parse_args()

    names = args.tokenizers

    print("=" * 70)
    print(f"COMPRESSION ON PRE-TOKENIZED DATA (split={args.split})")
    print("=" * 70)

    results = {}
    for name in names:
        ds = load_from_disk(f"{DATA_PATHS[name]}/{args.split}")
        total_tokens = sum(len(x) for x in ds["input_ids"])
        total_docs = len(ds)
        results[name] = {"total_tokens": total_tokens, "docs": total_docs,
                         "avg_tokens": total_tokens / total_docs}
        print(f"\n  {SHORT[name]}:")
        print(f"    Documents:    {total_docs:>12,}")
        print(f"    Total tokens: {total_tokens:>12,}")
        print(f"    Avg tok/doc:  {results[name]['avg_tokens']:>12.1f}")

    if args.baseline in names:
        baseline_tokens = results[args.baseline]["total_tokens"]
        print(f"\n  Relative to {SHORT[args.baseline]}:")
        for name in names:
            diff = (results[name]["total_tokens"] - baseline_tokens) / baseline_tokens * 100
            abs_diff = abs(results[name]["total_tokens"] - baseline_tokens)
            direction = "more" if diff > 0 else "fewer"
            print(f"    {SHORT[name]:6s}: {diff:+.3f}%  ({abs_diff:,} {direction} tokens)")

    if args.sample_texts > 0:
        print(f"\n{'=' * 70}")
        print(f"LIVE TOKENIZATION ({args.sample_texts} raw texts from test)")
        print("=" * 70)

        raw_data = load_from_disk(RAW_TEST_PATH)
        sample = raw_data.select(range(min(args.sample_texts, len(raw_data))))
        tokenizers = {n: AutoTokenizer.from_pretrained(TOKENIZER_PATHS[n]) for n in names}

        live_results = {n: 0 for n in names}
        for text in sample["text"]:
            for name in names:
                ids = tokenizers[name].encode(text, add_special_tokens=False)
                live_results[name] += len(ids)

        for name in names:
            print(f"  {SHORT[name]:6s}: {live_results[name]:>10,} tokens")

        if args.baseline in names:
            bl = live_results[args.baseline]
            print(f"\n  Relative to {SHORT[args.baseline]}:")
            for name in names:
                diff = (live_results[name] - bl) / bl * 100
                print(f"    {SHORT[name]:6s}: {diff:+.3f}%")


if __name__ == "__main__":
    main()
