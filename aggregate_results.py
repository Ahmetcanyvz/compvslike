"""Aggregate BLiMP and BPB evaluation results across seeds.

Computes mean/std for each (tokenizer_type, vocab_size) combination.

Usage:
    uv run python aggregate_results.py [EVAL_RESULTS_DIR]
"""

import json
import math
import sys
from pathlib import Path

import polars as pl

EVAL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval_results")


def parse_model_folder(name: str) -> dict:
    """Parse folder name like me57M-tied_bpe-8k_15000steps_seed42."""
    parts = name.split("_")
    # parts: ['me57M-tied', 'bpe-8k', '15000steps', 'seed42']
    # But tokenizer name can have dashes and underscores, e.g. greedyll-exact-8k
    # Strategy: strip known prefix and suffix
    rest = name.removeprefix("me57M-tied_")
    # Find seed
    seed_part = [p for p in rest.split("_") if p.startswith("seed")][-1]
    seed = int(seed_part.replace("seed", ""))
    # Find steps
    steps_part = [p for p in rest.split("_") if p.endswith("steps")][-1]
    # Tokenizer is everything before _NNNNNsteps
    idx = rest.index(f"_{steps_part}")
    tok_full = rest[:idx]  # e.g. bpe-8k, compmax-32k, greedyll-exact-128k
    # Split tok_full into type and vocab: last -Nk part is vocab
    last_dash = tok_full.rfind("-")
    tok_type = tok_full[:last_dash]  # bpe, compmax, greedyll-exact
    vocab_size = tok_full[last_dash + 1:]  # 8k, 32k, 128k
    return {
        "model_folder": name,
        "tok_type": tok_type,
        "vocab_size": vocab_size,
        "seed": seed,
    }


def load_blimp_accuracy(model_dir: Path) -> float | None:
    """Load overall BLiMP accuracy from summary JSON."""
    summary = model_dir / "blimp.summary.json"
    if summary.exists():
        with open(summary) as f:
            return json.load(f)["overall_accuracy"]
    # Fallback: compute from parquet
    parquet = model_dir / "blimp.parquet"
    if parquet.exists():
        df = pl.read_parquet(parquet)
        return df["correct"].mean()
    return None


def load_bpb(model_dir: Path) -> dict | None:
    """Load aggregate BPB and perplexity from parquet."""
    parquet = model_dir / "bpb.parquet"
    if not parquet.exists():
        return None
    df = pl.read_parquet(parquet)
    total_loss = df["loss_nats"].sum()
    total_bytes = df["num_bytes"].sum()
    total_tokens = df["num_tokens"].sum()
    aggregate_bpb = total_loss / (total_bytes * math.log(2))
    perplexity = math.exp(total_loss / total_tokens)
    return {
        "bpb": aggregate_bpb,
        "perplexity": perplexity,
        "bytes_per_token": total_bytes / total_tokens,
    }


def main():
    if not EVAL_DIR.exists():
        print(f"Error: {EVAL_DIR} not found")
        sys.exit(1)

    rows = []
    for model_dir in sorted(EVAL_DIR.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("me57M-tied_"):
            continue

        info = parse_model_folder(model_dir.name)
        blimp_acc = load_blimp_accuracy(model_dir)
        bpb_info = load_bpb(model_dir)

        row = {**info}
        row["blimp_accuracy"] = blimp_acc
        if bpb_info:
            row.update(bpb_info)
        else:
            row["bpb"] = None
            row["perplexity"] = None
            row["bytes_per_token"] = None

        rows.append(row)

    if not rows:
        print("No results found.")
        sys.exit(1)

    df = pl.DataFrame(rows)

    # ── Per-model results ─────────────────────────────────────────────────
    print("=" * 80)
    print("PER-MODEL RESULTS")
    print("=" * 80)
    print(df.select("tok_type", "vocab_size", "seed", "blimp_accuracy", "bpb", "perplexity", "bytes_per_token"))

    # ── Aggregated: mean ± std across seeds ───────────────────────────────
    agg = (
        df.group_by("tok_type", "vocab_size")
        .agg(
            pl.col("blimp_accuracy").mean().alias("blimp_mean"),
            pl.col("blimp_accuracy").std().alias("blimp_std"),
            pl.col("bpb").mean().alias("bpb_mean"),
            pl.col("bpb").std().alias("bpb_std"),
            pl.col("perplexity").mean().alias("ppl_mean"),
            pl.col("perplexity").std().alias("ppl_std"),
            pl.col("bytes_per_token").mean().alias("bpt_mean"),
            pl.col("seed").count().alias("n_seeds"),
        )
        .sort("tok_type", "vocab_size")
    )

    print("\n" + "=" * 80)
    print("AGGREGATED (mean ± std across seeds)")
    print("=" * 80)

    # Print formatted table
    header = f"{'Tokenizer':<20} {'Vocab':>6} {'N':>3} {'BLiMP Acc':>16} {'BPB':>16} {'Perplexity':>18} {'Bytes/Tok':>10}"
    print(header)
    print("-" * len(header))

    for row in agg.iter_rows(named=True):
        blimp_str = f"{row['blimp_mean']:.3f} ± {row['blimp_std']:.3f}" if row["blimp_mean"] is not None else "N/A"
        bpb_str = f"{row['bpb_mean']:.4f} ± {row['bpb_std']:.4f}" if row["bpb_mean"] is not None else "N/A"
        ppl_str = f"{row['ppl_mean']:.2f} ± {row['ppl_std']:.2f}" if row["ppl_mean"] is not None else "N/A"
        bpt_str = f"{row['bpt_mean']:.2f}" if row["bpt_mean"] is not None else "N/A"
        print(f"{row['tok_type']:<20} {row['vocab_size']:>6} {row['n_seeds']:>3} {blimp_str:>16} {bpb_str:>16} {ppl_str:>18} {bpt_str:>10}")

    # ── Save to CSV ───────────────────────────────────────────────────────
    out_per_model = EVAL_DIR / "per_model_results.csv"
    out_aggregated = EVAL_DIR / "aggregated_results.csv"
    df.write_csv(out_per_model)
    agg.write_csv(out_aggregated)
    print(f"\nSaved: {out_per_model}")
    print(f"Saved: {out_aggregated}")


if __name__ == "__main__":
    main()
