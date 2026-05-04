#!/usr/bin/env python3
"""Analyze merge overlap and ordering differences across BPE-based tokenizer methods.

Note: Only applicable to BPE-based methods (BPE, GreedyLL-Exact, GreedyLL-Approx).
Unigram methods do not use merges.
"""

import argparse
import json
from config import TOKENIZER_PATHS, SHORT, BPE_METHODS


def load_merges(path):
    with open(f"{path}/tokenizer.json") as f:
        tj = json.load(f)
    merge_list = tj["model"]["merges"]
    normalized = []
    for m in merge_list:
        if isinstance(m, list):
            normalized.append(tuple(m))
        else:
            normalized.append(tuple(m.split()))
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Analyze merge overlap across BPE methods")
    parser.add_argument("--tokenizers", nargs="+", default=BPE_METHODS,
                        choices=BPE_METHODS, help="BPE tokenizer names to compare")
    parser.add_argument("--rank-window", type=int, default=1000)
    parser.add_argument("--show-early-diffs", type=int, default=20)
    args = parser.parse_args()

    names = args.tokenizers
    merges = {name: load_merges(TOKENIZER_PATHS[name]) for name in names}
    merge_sets = {name: set(m) for name, m in merges.items()}

    # ---- Pairwise overlap ----
    print("=" * 70)
    print("MERGE OVERLAP (BPE methods only)")
    print("=" * 70)

    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            shared = merge_sets[n1] & merge_sets[n2]
            only1 = merge_sets[n1] - merge_sets[n2]
            only2 = merge_sets[n2] - merge_sets[n2]
            total = len(merge_sets[n1])
            print(f"\n{SHORT[n1]} vs {SHORT[n2]}:")
            print(f"  Total merges:  {total:>7,}")
            print(f"  Shared:        {len(shared):>7,}  ({len(shared)/total*100:.1f}%)")
            print(f"  Only {SHORT[n1]:6s}:  {len(only1):>7,}")
            print(f"  Only {SHORT[n2]:6s}:  {len(merge_sets[n2] - merge_sets[n1]):>7,}")

    # ---- Merge order agreement ----
    print(f"\n{'=' * 70}")
    print(f"MERGE ORDER AGREEMENT (first {args.rank_window} merges)")
    print("=" * 70)

    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            shared = merge_sets[n1] & merge_sets[n2]
            rank1 = {m: idx for idx, m in enumerate(merges[n1]) if m in shared}
            rank2 = {m: idx for idx, m in enumerate(merges[n2]) if m in shared}

            window = args.rank_window
            early1 = [m for m in merges[n1][:window] if m in shared]
            if early1:
                rank_diffs = [abs(rank1[m] - rank2[m]) for m in early1]
                avg_diff = sum(rank_diffs) / len(rank_diffs)
                same_pos = sum(1 for d in rank_diffs if d == 0)
                within_10 = sum(1 for d in rank_diffs if d <= 10)
                within_100 = sum(1 for d in rank_diffs if d <= 100)

                print(f"\n{SHORT[n1]} vs {SHORT[n2]} (first {window} merges of {SHORT[n1]}):")
                print(f"  Shared in window:  {len(early1)}")
                print(f"  Same position:     {same_pos}")
                print(f"  Within ±10 ranks:  {within_10}")
                print(f"  Within ±100 ranks: {within_100}")
                print(f"  Avg rank diff:     {avg_diff:.1f}")

    # ---- Side by side ----
    print(f"\n{'=' * 70}")
    print(f"FIRST {args.show_early_diffs} MERGES — SIDE BY SIDE")
    print("=" * 70)

    max_show = args.show_early_diffs
    header = f"  {'#':>4}"
    for name in names:
        header += f"  {SHORT[name]:>20}"
    print(header)
    print("  " + "-" * (4 + 22 * len(names)))

    for i in range(max_show):
        row = f"  {i+1:>4}"
        for name in names:
            if i < len(merges[name]):
                m = merges[name][i]
                row += f"  {f'{m[0]} + {m[1]}':>20}"
            else:
                row += f"  {'—':>20}"
        print(row)

    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            for k in range(min(len(merges[n1]), len(merges[n2]))):
                if merges[n1][k] != merges[n2][k]:
                    print(f"\n{SHORT[n1]} vs {SHORT[n2]}: first divergence at merge #{k+1}")
                    print(f"  {SHORT[n1]}: {merges[n1][k][0]} + {merges[n1][k][1]}")
                    print(f"  {SHORT[n2]}: {merges[n2][k][0]} + {merges[n2][k][1]}")
                    break


if __name__ == "__main__":
    main()
