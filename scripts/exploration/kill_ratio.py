#!/usr/bin/env python3
"""
Kill ratio analysis: how many merge results are just stepping stones?

Only applicable to BPE-based methods (BPE, GreedyLL-Exact, GreedyLL-Approx).
"""

import argparse
import json
from collections import Counter, defaultdict
from datasets import load_from_disk
from config import TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS


def load_merges(path):
    with open(f"{path}/tokenizer.json") as f:
        tj = json.load(f)
    merge_list = tj["model"]["merges"]
    triples = []
    for m in merge_list:
        if isinstance(m, list):
            left, right = m[0], m[1]
        else:
            left, right = m.split(" ", 1)
        triples.append((left, right, left + right))
    return triples


def analyze_kill_ratio(merges, min_lengths=(0, 3, 5, 10)):
    merge_results = set()
    killed = set()
    for left, right, result in merges:
        if left in merge_results:
            killed.add(left)
        if right in merge_results:
            killed.add(right)
        merge_results.add(result)
    survived = merge_results - killed

    results = {}
    for min_len in min_lengths:
        fc = {t for t in merge_results if len(t) >= min_len}
        fk = {t for t in killed if len(t) >= min_len}
        fs = {t for t in survived if len(t) >= min_len}
        results[min_len] = {"created": len(fc), "killed": len(fk), "survived": len(fs),
                            "ratio": len(fk)/len(fc)*100 if fc else 0}
    return results, killed, survived


def find_longest_chains(merges):
    merge_results = set()
    merged_into = {}
    for left, right, result in merges:
        if left in merge_results and left not in merged_into:
            merged_into[left] = result
        if right in merge_results and right not in merged_into:
            merged_into[right] = result
        merge_results.add(result)

    chains = []
    for _, _, result in merges:
        chain = [result]
        current = result
        while current in merged_into:
            current = merged_into[current]
            chain.append(current)
        if len(chain) > 1:
            chains.append(chain)

    chains.sort(key=len, reverse=True)
    seen = set()
    maximal = []
    for chain in chains:
        if chain[0] not in seen:
            maximal.append(chain)
            seen.update(chain)
    return maximal


def main():
    parser = argparse.ArgumentParser(description="Kill ratio analysis (BPE methods only)")
    parser.add_argument("--tokenizers", nargs="+", default=BPE_METHODS,
                        choices=BPE_METHODS)
    parser.add_argument("--min-lengths", type=int, nargs="+", default=[0, 3, 5, 10])
    parser.add_argument("--top-chains", type=int, default=15)
    parser.add_argument("--check-usage", action="store_true",
                        help="Cross-reference with tokenized test data")
    args = parser.parse_args()

    names = args.tokenizers
    all_killed = {}
    all_survived = {}

    print("=" * 70)
    print("KILL RATIO ANALYSIS (BPE methods only)")
    print("=" * 70)

    for name in names:
        merges = load_merges(TOKENIZER_PATHS[name])
        results, killed, survived = analyze_kill_ratio(merges, args.min_lengths)
        all_killed[name] = killed
        all_survived[name] = survived

        print(f"\n{SHORT[name]}:")
        print(f"  {'Min len':>8}  {'Created':>10}  {'Killed':>10}  {'Survived':>10}  {'Kill %':>8}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        for ml in args.min_lengths:
            r = results[ml]
            print(f"  {ml:>8}  {r['created']:>10,}  {r['killed']:>10,}  {r['survived']:>10,}  {r['ratio']:>7.1f}%")

    # ---- Longest chains ----
    print(f"\n{'=' * 70}")
    print(f"LONGEST MERGE CHAINS (top {args.top_chains})")
    print("=" * 70)

    for name in names:
        merges = load_merges(TOKENIZER_PATHS[name])
        chains = find_longest_chains(merges)
        print(f"\n{SHORT[name]}:")
        for i, chain in enumerate(chains[:args.top_chains]):
            print(f"  {i+1:>3}. (len {len(chain):>2}) {' → '.join(chain)}")

    # ---- Usage check ----
    if args.check_usage:
        print(f"\n{'=' * 70}")
        print("ACTUAL USAGE OF KILLED TOKENS IN TEST SET")
        print("=" * 70)

        for name in names:
            with open(f"{TOKENIZER_PATHS[name]}/tokenizer.json") as f:
                tj = json.load(f)
            vocab = tj["model"]["vocab"]
            id_to_token = {v: k for k, v in vocab.items()}

            ds = load_from_disk(f"{DATA_PATHS[name]}/test")
            usage = Counter()
            for doc_ids in ds["input_ids"]:
                usage.update(doc_ids)

            killed = all_killed[name]
            killed_ids = {vocab[t] for t in killed if t in vocab}
            zero = sum(1 for tid in killed_ids if usage[tid] == 0)
            low = sum(1 for tid in killed_ids if usage[tid] <= 10)
            any_use = sum(1 for tid in killed_ids if usage[tid] > 0)

            print(f"\n{SHORT[name]} ({len(killed_ids):,} killed tokens):")
            print(f"  Zero usage:  {zero:>7,} ({zero/len(killed_ids)*100:.1f}%)")
            print(f"  Low (≤10):   {low:>7,} ({low/len(killed_ids)*100:.1f}%)")
            print(f"  Has usage:   {any_use:>7,} ({any_use/len(killed_ids)*100:.1f}%)")

            killed_with_usage = [(id_to_token[tid], usage[tid])
                                 for tid in killed_ids if usage[tid] > 0 and tid in id_to_token]
            killed_with_usage.sort(key=lambda x: -x[1])
            print(f"  Top used 'killed' tokens:")
            for token, count in killed_with_usage[:10]:
                print(f"    {token!r:30s}  used {count:>8,} times")


if __name__ == "__main__":
    main()
