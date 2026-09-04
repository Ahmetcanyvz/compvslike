#!/usr/bin/env python3
"""Analyze token length distributions across tokenizer methods."""

import argparse
from collections import Counter
from config import TOKENIZER_PATHS, SHORT, ALL_METHODS, load_vocab


def main():
    parser = argparse.ArgumentParser(description="Token length distribution analysis")
    parser.add_argument("--tokenizers", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Tokenizer names to compare")
    parser.add_argument("--max-display-len", type=int, default=15)
    parser.add_argument("--bar-scale", type=int, default=500)
    args = parser.parse_args()

    names = args.tokenizers
    all_counts = {}

    print("=" * 70)
    print("TOKEN LENGTH DISTRIBUTION (in characters)")
    print("=" * 70)

    for name in names:
        vocab = load_vocab(TOKENIZER_PATHS[name])
        tokens = list(vocab.keys())
        lengths = [len(t) for t in tokens]
        avg = sum(lengths) / len(lengths)
        length_counts = Counter(lengths)
        all_counts[name] = length_counts

        print(f"\n{SHORT[name]} (vocab={len(tokens):,}):")
        print(f"  Avg token length: {avg:.2f} chars")
        print(f"  Median:           {sorted(lengths)[len(lengths)//2]} chars")
        print(f"  Max:              {max(lengths)} chars")
        print(f"  Distribution:")

        for l in sorted(length_counts.keys()):
            if l <= args.max_display_len:
                bar = "#" * (length_counts[l] // args.bar_scale)
                print(f"    {l:3d} chars: {length_counts[l]:>6,}  {bar}")
            elif l == args.max_display_len + 1:
                remaining = sum(v for k, v in length_counts.items() if k > args.max_display_len)
                print(f"    {args.max_display_len+1}+ chars: {remaining:>6,}")

    # ---- Comparison table ----
    print(f"\n{'=' * 70}")
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Length':<10}", end="")
    for name in names:
        print(f"  {SHORT[name]:>10}", end="")
    print()
    print("  " + "-" * (10 + 12 * len(names)))

    for l in range(1, args.max_display_len + 1):
        print(f"  {l:<10}", end="")
        for name in names:
            print(f"  {all_counts[name].get(l, 0):>10,}", end="")
        print()

    print(f"  {f'{args.max_display_len+1}+':<10}", end="")
    for name in names:
        remaining = sum(v for k, v in all_counts[name].items() if k > args.max_display_len)
        print(f"  {remaining:>10,}", end="")
    print()


if __name__ == "__main__":
    main()
