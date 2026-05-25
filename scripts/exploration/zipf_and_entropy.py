#!/usr/bin/env python3
"""
Zipf's coefficient and entropy analysis across all tokenizer methods.

1. Zipf's coefficient (global and per-segment)
2. Shannon entropy and efficiency
3. Entropy contribution by frequency band
4. Cumulative coverage
"""

import argparse
import gc
import json
import math
import os
# Cap parallelism BEFORE importing tokenizers; on big login nodes the default
# rayon pool will happily spawn ~one worker per CPU (288+) and each one keeps
# scratch arenas that collectively run the process out of memory.
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import Counter
from pathlib import Path
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
from config import TOKENIZER_PATHS, DATA_PATHS, SHORT, ALL_METHODS, RAW_TEST_PATH, load_vocab


COUNTS_CACHE_DIR = Path(os.environ.get(
    "ZIPF_COUNTS_CACHE",
    str(Path(__file__).resolve().parent.parent.parent / "exploration_results" / "_counts_cache"),
))


def _cache_path(name, split):
    return COUNTS_CACHE_DIR / f"{name}__{split}.json"


def _save_counts(name, split, usage):
    COUNTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(name, split).with_suffix(".json.tmp")
    # JSON keys must be strings; store id->count.
    with open(tmp, "w") as f:
        json.dump({str(k): v for k, v in usage.items()}, f)
    tmp.replace(_cache_path(name, split))


def _load_counts(name, split):
    p = _cache_path(name, split)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return Counter({int(k): v for k, v in d.items()})


def _tokenize_and_save(name, split):
    """Stream the raw test split through this tokenizer and save a counts cache."""
    pre = Path(DATA_PATHS[name]) / split
    usage = Counter()
    if pre.exists():
        ds = load_from_disk(str(pre))
        for doc_ids in tqdm(ds["input_ids"], desc=f"  counting ({SHORT[name]}, pretok)", unit="doc"):
            usage.update(doc_ids)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"{TOKENIZER_PATHS[name]}/tokenizer.json")
        raw_ds = load_from_disk(RAW_TEST_PATH)
        text_col = "text" if "text" in raw_ds.column_names else raw_ds.column_names[0]
        n = len(raw_ds)
        BATCH = 256
        pbar = tqdm(total=n, desc=f"  tokenizing ({SHORT[name]}, raw)", unit="doc")
        batch = []
        for ex in raw_ds:
            batch.append(ex[text_col])
            if len(batch) >= BATCH:
                for e in tok.encode_batch(batch):
                    usage.update(e.ids)
                pbar.update(len(batch))
                batch.clear()
        if batch:
            for e in tok.encode_batch(batch):
                usage.update(e.ids)
            pbar.update(len(batch))
        pbar.close()
        del tok, raw_ds
        gc.collect()
    _save_counts(name, split, usage)
    return usage


def load_usage_counts(name, split, force=False):
    """Return a Counter of {token_id: count} on the test split.

    Uses a JSON counts cache to avoid re-tokenizing. The flow is:
      1. If cache exists (and not forced), load and return it.
      2. Otherwise tokenize, save the cache, free memory, and load fresh from disk.
    Tokenization frees the in-memory dataset and tokenizer before continuing so
    each call has a clean memory baseline."""
    if not force:
        cached = _load_counts(name, split)
        if cached is not None:
            return cached
    _tokenize_and_save(name, split)
    gc.collect()
    return _load_counts(name, split)


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
    parser.add_argument("--force", action="store_true",
                        help="Re-tokenize even if a counts cache exists.")
    args = parser.parse_args()

    names = args.tokenizers

    # Phase 1: tokenize + cache counts for every tokenizer.
    # Each iteration writes a counts JSON to disk and then drops
    # the in-memory dataset/tokenizer before moving to the next.
    for name in names:
        print(f"[tokenize] {SHORT[name]}...", flush=True)
        if not args.force and _cache_path(name, args.split).exists():
            print(f"  cache exists at {_cache_path(name, args.split)}, skipping", flush=True)
            continue
        _tokenize_and_save(name, args.split)
        gc.collect()

    # Phase 2: load cached counts back in (small, just id->count maps) and
    # compute the statistics.
    all_freqs = {}
    all_totals = {}
    all_vocab_sizes = {}

    for name in names:
        print(f"[load] {SHORT[name]}...", flush=True)
        usage = _load_counts(name, args.split)
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
