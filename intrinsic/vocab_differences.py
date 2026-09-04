#!/usr/bin/env python3
"""Compare vocabularies across all tokenizer methods."""

import argparse
from config import TOKENIZER_PATHS, SHORT, ALL_METHODS, load_vocab


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer vocabularies")
    parser.add_argument("--tokenizers", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Tokenizer names to compare")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of unique tokens to show per method")
    args = parser.parse_args()

    names = args.tokenizers
    vocabs = {name: set(load_vocab(TOKENIZER_PATHS[name]).keys()) for name in names}

    # ---- Pairwise overlap ----
    print("=" * 70)
    print("PAIRWISE VOCABULARY OVERLAP")
    print("=" * 70)

    header = f"  {'':>10}"
    for n2 in names:
        header += f"  {SHORT[n2]:>10}"
    print(header)
    print(f"  {'-'*10}" + f"  {'-'*10}" * len(names))
    for n1 in names:
        row = f"  {SHORT[n1]:<10}"
        for n2 in names:
            if n1 == n2:
                row += f"  {'—':>10}"
            else:
                overlap = len(vocabs[n1] & vocabs[n2]) / len(vocabs[n1]) * 100
                row += f"  {overlap:>9.1f}%"
        print(row)

    # ---- Tokens unique to each method ----
    print(f"\n{'=' * 70}")
    print(f"TOKENS UNIQUE TO EACH METHOD (top {args.top_n} by length)")
    print("=" * 70)

    for name in names:
        others = set()
        for other in names:
            if other != name:
                others |= vocabs[other]
        unique = vocabs[name] - others
        examples = sorted(list(unique), key=len, reverse=True)[:args.top_n]
        print(f"\n{SHORT[name]} ({len(unique):,} unique tokens):")
        for t in examples:
            print(f"  {t!r}")

    # ---- Shared across all ----
    all_shared = vocabs[names[0]]
    for name in names[1:]:
        all_shared &= vocabs[name]
    print(f"\n{'=' * 70}")
    print(f"Shared across ALL methods: {len(all_shared):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
