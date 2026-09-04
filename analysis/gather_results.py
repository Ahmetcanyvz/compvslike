"""Gather BPB and BLiMP results per model into a single JSON.

For each <model>/ directory under the root, reads bpb.parquet (byte-weighted
corpus BPB) and/or blimp.parquet (overall + per-task minimal-pair accuracy),
parses the model name into size/tokenizer/vocab/seed, and writes one JSON.

Usage:
    python scripts/gather_results.py eval_results_8k_32k -o eval_results_8k_32k/results.json
"""

import json
import math
import re
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

app = typer.Typer()
console = Console()
LN2 = math.log(2)

# me<size>M-tied_<tokenizer>-<vocab>_<N>Btok_seed<seed>
NAME_RE = re.compile(r"^me(?P<size>\d+M)-tied_(?P<tok>.+)-(?P<vocab>\d+k)_\d+Btok_seed(?P<seed>\d+)$")


def corpus_bpb(path: Path) -> tuple[float, int]:
    df = pl.read_parquet(path)
    loss = df["loss_nats"].to_numpy()
    nbytes = df["num_bytes"].to_numpy()
    return float(loss.sum() / (nbytes.sum() * LN2)), int(len(df))


def blimp_acc(path: Path) -> tuple[float, int, dict]:
    df = pl.read_parquet(path)
    overall = float(df["correct"].mean())
    per_task = (
        df.group_by("task").agg(pl.col("correct").mean().alias("acc"))
        .sort("task")
    )
    tasks = {r["task"]: round(float(r["acc"]), 4) for r in per_task.iter_rows(named=True)}
    return overall, int(len(df)), tasks


@app.command()
def main(
    root: Path = typer.Argument(..., help="Directory containing <model>/{bpb,blimp}.parquet"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON path"),
    per_task: bool = typer.Option(True, "--per-task/--no-per-task", help="Include BLiMP per-task accuracy"),
) -> None:
    model_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    results: dict[str, dict] = {}

    for d in model_dirs:
        m = NAME_RE.match(d.name)
        if not m:
            continue
        entry: dict = {
            "size": m.group("size"),
            "tokenizer": m.group("tok"),
            "vocab": m.group("vocab"),
            "seed": int(m.group("seed")),
        }
        bpb_f = d / "bpb.parquet"
        blimp_f = d / "blimp.parquet"
        if bpb_f.exists():
            entry["bpb"], entry["bpb_n_docs"] = corpus_bpb(bpb_f)
            entry["bpb"] = round(entry["bpb"], 4)
        if blimp_f.exists():
            acc, npairs, tasks = blimp_acc(blimp_f)
            entry["blimp_accuracy"] = round(acc, 4)
            entry["blimp_n_pairs"] = npairs
            if per_task:
                entry["blimp_per_task"] = tasks
        if "bpb" in entry or "blimp_accuracy" in entry:
            results[d.name] = entry

    if not results:
        console.print(f"[red]No parseable model results found under {root}[/red]")
        raise typer.Exit(1)

    # Console summary sorted by (size, vocab, bpb)
    console.print(f"[green]Gathered {len(results)} models[/green]")
    for name, e in sorted(results.items(), key=lambda kv: (kv[1]["size"], kv[1]["vocab"], kv[1].get("bpb", 9))):
        bpb = f"{e['bpb']:.4f}" if "bpb" in e else "  —  "
        bl = f"{e['blimp_accuracy']:.4f}" if "blimp_accuracy" in e else "  —  "
        console.print(f"  {e['size']:>5} {e['vocab']:>4} {e['tokenizer']:<16} BPB={bpb}  BLiMP={bl}")

    if output is None:
        output = root / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Wrote {output}[/green]")


if __name__ == "__main__":
    app()
