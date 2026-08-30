"""Plot Zipf rank-frequency curves with fitted power laws.

Replaces the table of Zipf alpha values with a figure: for each tokeniser we
plot the empirical rank-frequency curve on log-log axes, overlaid with the
fitted line whose slope is -alpha.

Reuses the token-count caches written by zipf_and_entropy.py, so no
re-tokenisation is needed if those already exist.

Usage:
    python scripts/exploration/zipf_plot.py                       # english, all vocab sizes
    python scripts/exploration/zipf_plot.py --vocab-sizes 8k 32k 128k
    EXPLORATION_VARIANT=multi python scripts/exploration/zipf_plot.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SHORT  # noqa: E402
from zipf_and_entropy import fit_zipf, load_usage_counts  # noqa: E402

# One colour per method; vocab size varies by panel.
METHOD_COLOURS = {
    "bpe": "#1f77b4",
    "compmax": "#d62728",
    "greedyll-exact": "#2ca02c",
    "unigramlm": "#9467bd",
    "greedyll-approx": "#8c564b",
}
METHOD_LABELS = {
    "bpe": "BPE",
    "compmax": "CompMax",
    "greedyll-exact": "GreedyLL",
    "unigramlm": "UnigramLM",
    "greedyll-approx": "GreedyLL (approx)",
}


def rank_freq(counts):
    """Sorted non-zero frequencies and their ranks."""
    f = np.array(sorted(counts, reverse=True), dtype=np.float64)
    f = f[f > 0]
    return np.arange(1, len(f) + 1, dtype=np.float64), f


def main():
    p = argparse.ArgumentParser(description="Zipf rank-frequency figure")
    p.add_argument("--methods", nargs="+",
                   default=["bpe", "compmax", "greedyll-exact", "unigramlm"])
    p.add_argument("--vocab-sizes", nargs="+", default=["8k", "32k", "128k"])
    p.add_argument("--split", default="test")
    p.add_argument("--output", "-o", default="zipf_curves.pdf")
    p.add_argument("--subsample", type=int, default=2000,
                   help="Points to draw per curve (log-spaced); 0 = all")
    args = p.parse_args()

    n = len(args.vocab_sizes)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, vs in zip(axes, args.vocab_sizes):
        for method in args.methods:
            name = f"{method}-{vs}"
            try:
                counts = load_usage_counts(name, args.split)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {name}: {e}")
                continue

            ranks, freqs = rank_freq(list(counts.values()))
            if len(freqs) == 0:
                print(f"  [skip] {name}: no non-zero counts")
                continue

            alpha, r2 = fit_zipf(list(counts.values()))
            colour = METHOD_COLOURS.get(method, None)
            label = f"{METHOD_LABELS.get(method, method)} ($\\alpha$={alpha:.2f})"

            # Log-spaced subsample keeps the PDF small without changing the shape.
            if args.subsample and len(ranks) > args.subsample:
                idx = np.unique(np.geomspace(1, len(ranks), args.subsample).astype(int)) - 1
                r_plot, f_plot = ranks[idx], freqs[idx]
            else:
                r_plot, f_plot = ranks, freqs

            ax.plot(r_plot, f_plot, lw=1.2, color=colour, label=label, alpha=0.85)

            # Fitted power law: log f = c - alpha * log r, anchored at the empirical median.
            c = np.median(np.log(freqs) + alpha * np.log(ranks))
            ax.plot(ranks, np.exp(c - alpha * np.log(ranks)),
                    ls="--", lw=0.9, color=colour, alpha=0.55)
            print(f"  {name:<24} alpha={alpha:.4f}  R2={r2:.4f}  types={len(freqs):,}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_title(f"{vs} vocabulary")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.2, lw=0.4)

    axes[0].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
