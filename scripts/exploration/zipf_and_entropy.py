#!/usr/bin/env python3
"""
Zipf's coefficient and entropy analysis across all tokenizer methods.

1. Zipf's coefficient (global and per-segment)
2. Shannon entropy and efficiency
3. Entropy contribution by frequency band
4. Cumulative coverage
"""

import argparse
import math
import json
from collections import Counter
from pathlib import Path
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
from config import TOKENIZER_PATHS, DATA_PATHS, SHORT, ALL_METHODS, RAW_TEST_PATH, load_vocab


def load_usage_counts(name, split):
    """Return a Counter of {token_id: count} on the test split.

    Prefers pre-tokenized data at DATA_PATHS[name]/<split>. Falls back to
    tokenizing the raw text at RAW_TEST_PATH on the fly using the tokenizer's
    AutoTokenizer for tokenizers without pre-tokenized data (e.g., 8k/32k)."""
    pre = Path(DATA_PATHS[name]) / split
    usage = Counter()
    if pre.exists():
        ds = load_from_disk(str(pre))
        for doc_ids in tqdm(ds["input_ids"], desc=f"  counting ({SHORT[name]}, pretok)", unit="doc"):
            usage.update(doc_ids)
        return usage

    # Fallback: tokenize raw text with this tokenizer via the low-level
    # tokenizers library directly (avoids transformers' AutoTokenizer, which
    # can pull in protobuf/sentencepiece for some configs).
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{TOKENIZER_PATHS[name]}/tokenizer.json")
    raw_ds = load_from_disk(RAW_TEST_PATH)
    text_col = "text" if "text" in raw_ds.column_names else raw_ds.column_names[0]
    BATCH = 1024
    docs = raw_ds[text_col]
    for i in tqdm(range(0, len(docs), BATCH), desc=f"  tokenizing ({SHORT[name]}, raw)", unit="batch"):
        encs = tok.encode_batch(docs[i:i + BATCH])
        for e in encs:
            usage.update(e.ids)
    return usage


def fit_zipf(freq_values):
    """Fit Zipf's law via log-log regression. Returns (alpha, R²)."""
    freqs = np.array(sorted(freq_values, reverse=True), dtype=np.float64)
    freqs = freqs[freqs > 0]
    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    log_r, log_f = np.log(ranks), np.log(freqs)
    n = len(log_r)
    sx, sy, sxy, sx2 = log_r.sum(), log_f.sum(), (log_r * log_f).sum(), (log_r ** 2).sum()
    slope = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
    intercept = (sy - slope * sx) / n
    y_pred = intercept + slope * log_r
    ss_res = ((log_f - y_pred) ** 2).sum()
    ss_tot = ((log_f - log_f.mean()) ** 2).sum()
    return -slope, 1 - ss_res / ss_tot


def fit_zipf_segment(freq_values, start, end):
    """Fit Zipf on a rank segment [start, end)."""
    freqs = np.array(sorted(freq_values, reverse=True), dtype=np.float64)
    freqs = freqs[freqs > 0]
    end = min(end, len(freqs))
    if start >= end:
        return None, None
    seg_f = freqs[start:end]
    seg_r = np.arange(start + 1, end + 1, dtype=np.float64)
    log_r, log_f = np.log(seg_r), np.log(seg_f)
    n = len(log_r)
    sx, sy, sxy, sx2 = log_r.sum(), log_f.sum(), (log_r * log_f).sum(), (log_r ** 2).sum()
    d = n * sx2 - sx ** 2
    if d == 0:
        return None, None
    slope = (n * sxy - sx * sy) / d
    intercept = (sy - slope * sx) / n
    y_pred = intercept + slope * log_r
    ss_res = ((log_f - y_pred) ** 2).sum()
    ss_tot = ((log_f - log_f.mean()) ** 2).sum()
    return -slope, 1 - ss_res / ss_tot if ss_tot > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Zipf and entropy analysis")
    parser.add_argument("--tokenizers", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    names = args.tokenizers

    # Load frequencies
    all_freqs = {}
    all_totals = {}
    all_vocab_sizes = {}

    for name in names:
        print(f"Loading {SHORT[name]}...", flush=True)
        usage = load_usage_counts(name, args.split)
        all_freqs[name] = list(usage.values())
        all_totals[name] = sum(usage.values())
        all_vocab_sizes[name] = len(load_vocab(TOKENIZER_PATHS[name]))

    # ================================================================
    # 1. ZIPF'S COEFFICIENT
    # ================================================================
    print("\n" + "=" * 80)
    print("1. ZIPF'S COEFFICIENT (frequency ∝ 1/rank^α)")
    print("=" * 80)

    print(f"\n  Global fit:")
    print(f"  {'Method':<10}  {'α':>8}  {'R²':>8}  {'Non-zero':>10}  {'Zero':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    for name in names:
        alpha, r2 = fit_zipf(all_freqs[name])
        n_nz = sum(1 for f in all_freqs[name] if f > 0)
        n_z = all_vocab_sizes[name] - n_nz
        print(f"  {SHORT[name]:<10}  {alpha:>8.4f}  {r2:>8.4f}  {n_nz:>10,}  {n_z:>8,}")

    segments = [(1, 100), (100, 1000), (1000, 10000), (10000, 50000), (50000, 128000)]
    print(f"\n  α by rank segment:")
    header = f"  {'Rank range':<18}"
    for name in names:
        header += f"  {SHORT[name]:>10}"
    print(header)
    print(f"  {'-'*18}" + f"  {'-'*10}" * len(names))
    for start, end in segments:
        row = f"  {f'{start:,}-{end:,}':<18}"
        for name in names:
            alpha, _ = fit_zipf_segment(all_freqs[name], start - 1, end)
            row += f"  {alpha:>10.3f}" if alpha is not None else f"  {'n/a':>10}"
        print(row)

    # ================================================================
    # 2. SHANNON ENTROPY
    # ================================================================
    print("\n" + "=" * 80)
    print("2. SHANNON ENTROPY  H = -Σ P(token) · log₂(P(token))")
    print("=" * 80)

    print(f"\n  {'Method':<10}  {'Entropy':>10}  {'Used tokens':>12}  {'Efficiency':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")
    for name in names:
        total = all_totals[name]
        H = sum(-f / total * math.log2(f / total) for f in all_freqs[name] if f > 0)
        n_used = sum(1 for f in all_freqs[name] if f > 0)
        H_max = math.log2(n_used)
        eff = H / H_max * 100
        print(f"  {SHORT[name]:<10}  {H:>10.4f}  {n_used:>12,}  {eff:>11.2f}%")

    # ================================================================
    # 3. ENTROPY BY FREQUENCY BAND
    # ================================================================
    print("\n" + "=" * 80)
    print("3. ENTROPY CONTRIBUTION BY FREQUENCY BAND")
    print("=" * 80)

    bands = [("Top 100", 0, 100), ("101-1k", 100, 1000), ("1k-10k", 1000, 10000),
             ("10k-50k", 10000, 50000), ("50k-128k", 50000, 128000)]

    # Header
    print(f"\n  {'Band':<12}", end="")
    for name in names:
        print(f"  {SHORT[name]+' %corp':>12}  {SHORT[name]+' %H':>10}", end="")
    print()
    print(f"  {'-'*12}" + (f"  {'-'*12}  {'-'*10}") * len(names))

    for label, start, end in bands:
        print(f"  {label:<12}", end="")
        for name in names:
            freqs_sorted = sorted(all_freqs[name], reverse=True)
            total = all_totals[name]
            H_total = sum(-f / total * math.log2(f / total) for f in freqs_sorted if f > 0)
            seg = [f for f in freqs_sorted[start:min(end, len(freqs_sorted))] if f > 0]
            seg_sum = sum(seg)
            pct_corpus = seg_sum / total * 100
            H_band = sum(-f / total * math.log2(f / total) for f in seg)
            pct_H = H_band / H_total * 100
            print(f"  {pct_corpus:>11.2f}%  {pct_H:>9.2f}%", end="")
        print()

    # ================================================================
    # 4. CUMULATIVE COVERAGE
    # ================================================================
    print("\n" + "=" * 80)
    print("4. CUMULATIVE COVERAGE — tokens needed for X% of corpus")
    print("=" * 80)

    thresholds = [50, 75, 90, 95, 99, 99.9]
    header = f"\n  {'Coverage':<12}"
    for name in names:
        header += f"  {SHORT[name]:>10}"
    print(header)
    print(f"  {'-'*12}" + f"  {'-'*10}" * len(names))

    for thr in thresholds:
        row = f"  {thr}%{'':<9}"
        for name in names:
            freqs_sorted = sorted(all_freqs[name], reverse=True)
            total = all_totals[name]
            target = total * thr / 100
            cumsum = 0
            for i, f in enumerate(freqs_sorted):
                cumsum += f
                if cumsum >= target:
                    row += f"  {i + 1:>10,}"
                    break
        print(row)


if __name__ == "__main__":
    main()
