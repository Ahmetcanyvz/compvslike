"""Per-tokenizer BPB mean +/- std across seeds (multi-seed summary).

For each bpb.parquet it computes the byte-weighted corpus BPB:
    BPB = sum(loss_nats) / (sum(num_bytes) * ln2)
then groups by tokenizer and reports mean, std, and the per-seed values across
seeds (e.g. 42/43/44). Reports the standard deviation (std, ddof=1) across
seeds -- the same "mean +/- std" convention the paper uses (never variance).

Usage:
    python scripts/bpb_seed_std.py eval_results_1B
    python scripts/bpb_seed_std.py eval_results_1B --glob "me1B-tied_*-128k_20Btok_seed*/bpb.parquet" -o eval_results_1B/bpb_seed_std.json
"""

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()
LN2 = math.log(2)

# me1B-tied_<tokenizer>_20Btok_seed<seed>
NAME_RE = re.compile(r"^(?P<model>.+?)_(?P<tok>.+?)_\d+Btok_seed(?P<seed>\d+)$")


def corpus_bpb(path: Path) -> float:
    df = pl.read_parquet(path)
    loss = df["loss_nats"].to_numpy()
    nbytes = df["num_bytes"].to_numpy()
    return float(loss.sum() / (nbytes.sum() * LN2))


@app.command()
def main(
    root: Path = typer.Argument(..., help="Directory containing <model>/bpb.parquet dirs"),
    glob: str = typer.Option("*/bpb.parquet", "--glob"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="json/jsonl/csv/parquet by extension"),
) -> None:
    files = sorted(root.glob(glob))
    if not files:
        console.print(f"[red]No files matching {glob} under {root}[/red]")
        raise typer.Exit(1)

    # tokenizer -> {seed: bpb}
    by_tok: dict[str, dict[int, float]] = {}
    for f in files:
        name = f.parent.name
        m = NAME_RE.match(name)
        if not m:
            console.print(f"[yellow]Skipping unparseable dir: {name}[/yellow]")
            continue
        tok = m.group("tok")
        seed = int(m.group("seed"))
        by_tok.setdefault(tok, {})[seed] = corpus_bpb(f)

    rows = []
    for tok, seed_map in by_tok.items():
        seeds = sorted(seed_map)
        vals = np.array([seed_map[s] for s in seeds], dtype=np.float64)
        rows.append({
            "tokenizer": tok,
            "n_seeds": len(seeds),
            "mean_bpb": float(vals.mean()),
            "std_bpb": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min_bpb": float(vals.min()),
            "max_bpb": float(vals.max()),
            "seeds": ",".join(map(str, seeds)),
            "per_seed_bpb": ";".join(f"{s}:{seed_map[s]:.4f}" for s in seeds),
        })

    result = pl.DataFrame(rows).sort("mean_bpb")

    table = Table(title="1B BPB across seeds (byte-weighted corpus BPB)")
    table.add_column("tokenizer", overflow="fold")
    table.add_column("n", justify="right")
    table.add_column("mean BPB", justify="right")
    table.add_column("± std", justify="right")
    table.add_column("[min, max]", justify="right")
    table.add_column("per-seed", overflow="fold")
    for r in result.iter_rows(named=True):
        table.add_row(
            r["tokenizer"],
            str(r["n_seeds"]),
            f"{r['mean_bpb']:.4f}",
            f"{r['std_bpb']:.4f}",
            f"[{r['min_bpb']:.4f}, {r['max_bpb']:.4f}]",
            r["per_seed_bpb"],
        )
    console.print(table)

    # Warn about incomplete seed sets so a missing eval isn't silently averaged.
    max_n = max(r["n_seeds"] for r in rows)
    for r in rows:
        if r["n_seeds"] < max_n:
            console.print(f"[yellow]Note: {r['tokenizer']} has only {r['n_seeds']} seeds ({r['seeds']}).[/yellow]")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix == ".json":
            result.write_json(output)
        elif suffix in (".ndjson", ".jsonl"):
            result.write_ndjson(output)
        elif suffix == ".csv":
            result.write_csv(output)
        else:
            result.write_parquet(output)
        console.print(f"[green]Wrote summary to {output}[/green]")


if __name__ == "__main__":
    app()
