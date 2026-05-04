#!/usr/bin/env python3
"""Compare how different tokenizers split the same text."""

import argparse
from transformers import AutoTokenizer
from config import TOKENIZER_PATHS, SHORT, ALL_METHODS


DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The implementation of the algorithm was straightforward.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "人工智能正在改变世界",
    "https://www.example.com/path/to/resource?query=value&page=1",
    "Electroencephalography is used to diagnose neurological disorders.",
    "The United Nations was established on October 24, 1945.",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
    "¡Hola! ¿Cómo estás? Très bien, merci. Danke schön!",
]


def main():
    parser = argparse.ArgumentParser(description="Compare tokenization across methods")
    parser.add_argument("--text", type=str, nargs="+", default=None,
                        help="Custom text(s) to tokenize (uses defaults if omitted)")
    parser.add_argument("--tokenizers", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Tokenizer names to compare")
    parser.add_argument("--show-ids", action="store_true",
                        help="Also show token IDs")
    args = parser.parse_args()

    texts = args.text if args.text else DEFAULT_TEXTS
    names = args.tokenizers

    tokenizers = {n: AutoTokenizer.from_pretrained(TOKENIZER_PATHS[n]) for n in names}

    print("=" * 70)
    print("TOKENIZATION COMPARISON")
    print("=" * 70)

    for text in texts:
        print(f"\nText: {text!r}")
        print("-" * 70)

        results = {n: tokenizers[n].tokenize(text) for n in names}

        first = results[names[0]]
        all_same = all(results[n] == first for n in names[1:])

        for name in names:
            tokens = results[name]
            print(f"  {SHORT[name]:6s} ({len(tokens):2d} tok): {tokens}")
            if args.show_ids:
                ids = tokenizers[name].encode(text, add_special_tokens=False)
                print(f"  {' ':6s}    IDs : {ids}")

        if all_same:
            print("  --> All methods agree")
        else:
            counts = {SHORT[n]: len(results[n]) for n in names}
            best = min(counts, key=counts.get)
            print(f"  --> DIFFER! Fewest tokens: {best} ({counts[best]})")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    total_tokens = {SHORT[n]: 0 for n in names}
    differ_count = 0
    for text in texts:
        results = {n: tokenizers[n].tokenize(text) for n in names}
        for n in names:
            total_tokens[SHORT[n]] += len(results[n])
        if any(results[n] != results[names[0]] for n in names[1:]):
            differ_count += 1

    print(f"  Texts compared: {len(texts)}")
    print(f"  Texts where methods differ: {differ_count}")
    for label, total in total_tokens.items():
        print(f"  {label:6s} total tokens: {total}")


if __name__ == "__main__":
    main()
