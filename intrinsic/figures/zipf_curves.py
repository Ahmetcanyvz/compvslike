"""Zipf rank-frequency figure, generated locally.

Self-contained: tokenises the local raw test split with the local tokenisers,
counts token usage, fits a power law per tokeniser, and plots the empirical
rank-frequency curves with their fits on log-log axes.

Counts are cached under exploration_results/_counts_cache_local/ so reruns are fast.

Usage:
    .venv/bin/python scripts/exploration/zipf_plot_local.py -o zipf_curves.pdf
    .venv/bin/python scripts/exploration/zipf_plot_local.py --max-docs 20000 --vocab-sizes 128k
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "intrinsic" / "_counts_cache"

METHOD_COLOURS = {
    "bpe": "#1f77b4",
    "topdowncomp": "#d62728",
    "bottomupll-exact": "#2ca02c",
    "unigramlm": "#9467bd",
}
METHOD_LABELS = {
    "bpe": "BPE",
    "topdowncomp": "TopDownComp",
    "bottomupll-exact": "BottomUpLL",
    "unigramlm": "UnigramLM",
}


def fit_zipf(freqs):
    """Zipf fit via log-log regression. Returns (alpha, R^2)."""
    f = np.array(sorted(freqs, reverse=True), dtype=np.float64)
    f = f[f > 0]
    r = np.arange(1, len(f) + 1, dtype=np.float64)
    lr, lf = np.log(r), np.log(f)
    n = len(lr)
    sx, sy, sxy, sx2 = lr.sum(), lf.sum(), (lr * lf).sum(), (lr ** 2).sum()
    slope = (n * sxy - sx * sy) / (n * sx2 - sx ** 2)
    intercept = (sy - slope * sx) / n
    resid = ((lf - (intercept + slope * lr)) ** 2).sum()
    tot = ((lf - lf.mean()) ** 2).sum()
    return -slope, 1 - resid / tot


def get_counts(name, texts, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{name}__test.json"
    if cp.exists() and not force:
        with open(cp) as fh:
            return list(json.load(fh).values())

    from tokenizers import Tokenizer
    tok_path = ROOT / "tokenizers" / name / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(tok_path)
    tok = Tokenizer.from_file(str(tok_path))

    counter = Counter()
    B = 2000
    for i in range(0, len(texts), B):
        for enc in tok.encode_batch(texts[i:i + B]):
            counter.update(enc.ids)
        if (i // B) % 5 == 0:
            print(f"    {min(i + B, len(texts)):,}/{len(texts):,}", end="\r", flush=True)
    print(" " * 40, end="\r")

    with open(cp, "w") as fh:
        json.dump({str(k): v for k, v in counter.items()}, fh)
    return list(counter.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["bpe", "topdowncomp", "bottomupll-exact", "unigramlm"])
    p.add_argument("--vocab-sizes", nargs="+", default=["8k", "32k", "128k"])
    p.add_argument("--max-docs", type=int, default=20000,
                   help="Documents from the test split to use (0 = all 47,384)")
    p.add_argument("--data", default=str(ROOT / "data" / "fineweb-edu-raw" / "test"))
    p.add_argument("--output", "-o", default="zipf_curves.pdf")
    p.add_argument("--subsample", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--fontsize", type=float, default=12.0)
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": args.fontsize,
        "axes.labelsize": args.fontsize + 1,
        "axes.titlesize": args.fontsize + 1,
        "xtick.labelsize": args.fontsize - 1,
        "ytick.labelsize": args.fontsize - 1,
    })

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    if args.max_docs and args.max_docs < len(ds):
        ds = ds.select(range(args.max_docs))
    texts = ds["text"]
    print(f"Loaded {len(texts):,} documents from {args.data}\n")

    n = len(args.vocab_sizes)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, vs in zip(axes, args.vocab_sizes):
        print(f"[{vs}]")
        for method in args.methods:
            name = f"{method}-{vs}"
            try:
                counts = get_counts(name, texts, force=args.force)
            except FileNotFoundError:
                print(f"  [skip] {name}: tokenizer not found")
                continue

            f = np.array(sorted(counts, reverse=True), dtype=np.float64)
            f = f[f > 0]
            r = np.arange(1, len(f) + 1, dtype=np.float64)
            alpha, r2 = fit_zipf(counts)
            colour = METHOD_COLOURS.get(method)
            label = f"{METHOD_LABELS.get(method, method)} ($\\alpha$={alpha:.2f})"

            if args.subsample and len(r) > args.subsample:
                idx = np.unique(np.geomspace(1, len(r), args.subsample).astype(int)) - 1
                rp, fp = r[idx], f[idx]
            else:
                rp, fp = r, f

            ax.plot(rp, fp, lw=1.2, color=colour, label=label, alpha=0.85)
            c = np.median(np.log(f) + alpha * np.log(r))
            ax.plot(r, np.exp(c - alpha * np.log(r)), ls="--", lw=0.9, color=colour, alpha=0.55)
            print(f"  {name:<24} alpha={alpha:.4f}  R2={r2:.4f}  types={len(f):,}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_title(f"{vs} vocabulary")
        h, l = ax.get_legend_handles_labels()
        half = (len(h) + 1) // 2
        leg1 = ax.legend(h[:half], l[:half], loc="lower left",
                         fontsize=args.fontsize - 3, frameon=False,
                         handlelength=1.4, borderaxespad=0.3)
        ax.add_artist(leg1)
        ax.legend(h[half:], l[half:], loc="upper right",
                  fontsize=args.fontsize - 3, frameon=False,
                  handlelength=1.4, borderaxespad=0.3)
        ax.grid(alpha=0.2, lw=0.4)

    axes[0].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
