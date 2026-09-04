"""Vocabulary-overlap heatmap, replacing the pairwise overlap table.

Tokenisers are ordered by search procedure (bottom-up first, then top-down) so
that the block structure is visible: high overlap within each search family,
low overlap across families.

Usage:
    .venv/bin/python scripts/exploration/vocab_overlap_heatmap.py -o vocab_overlap_heatmap.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent

# Grouped by search procedure so the block structure shows up.
METHODS = [
    ("bpe", "BPE", "bottom-up"),
    ("bottomupll-exact", "BottomUpLL $=$", "bottom-up"),
    ("bottomupll-approx", "BottomUpLL $\\approx$", "bottom-up"),
    ("topdowncomp", "TopDownComp", "top-down"),
    ("unigramlm", "UnigramLM", "top-down"),
]


def load_vocab(method, size):
    with open(ROOT / "tokenizers" / f"{method}-{size}" / "tokenizer.json") as fh:
        tj = json.load(fh)
    v = tj["model"]["vocab"]
    return set(v.keys() if isinstance(v, dict) else [t[0] for t in v])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", default="128k")
    p.add_argument("--output", "-o", default="vocab_overlap_heatmap.pdf")
    p.add_argument("--fontsize", type=float, default=13.0)
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of method keys, in order (default: all five)")
    args = p.parse_args()

    methods = METHODS
    if args.methods:
        keep = {m: (m, l, f) for m, l, f in METHODS}
        methods = [keep[m] for m in args.methods]

    labels = [l for _, l, _ in methods]
    fams = [f for _, _, f in methods]
    V = {l: load_vocab(m, args.vocab_size) for m, l, _ in methods}

    n = len(labels)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                M[i, j] = 100 * len(V[a] & V[b]) / len(V[a])

    fs = args.fontsize
    fig, ax = plt.subplots(figsize=(1.45 * n + 2.0, 1.30 * n + 1.2))
    im = ax.imshow(M, cmap="Blues", vmin=np.nanmin(M), vmax=np.nanmax(M))

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", color="0.55", fontsize=fs)
            else:
                v = M[i, j]
                shade = "white" if v > (np.nanmin(M) + np.nanmax(M)) / 2 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color=shade, fontsize=fs)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)

    # Separator between the search-procedure blocks.
    if len(set(fams)) > 1:
        split = fams.index("top-down") - 0.5
        ax.axhline(split, color="0.15", lw=1.8)
        ax.axvline(split, color="0.15", lw=1.8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Vocabulary overlap (\\%)", fontsize=fs - 1)
    cb.ax.tick_params(labelsize=fs - 2)

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}\n")
    for i, a in enumerate(labels):
        print("  " + a.ljust(20) + " ".join(f"{M[i,j]:6.1f}" if i != j else "    --" for j in range(n)))


if __name__ == "__main__":
    main()
