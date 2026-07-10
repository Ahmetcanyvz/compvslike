"""Bootstrap confidence intervals for corpus-level bits-per-byte (BPB).

Corpus BPB is byte-weighted: sum(loss_nats) / (sum(num_bytes) * ln2).
We resample documents with replacement and recompute that aggregate each
iteration to get a CI that accounts for document-level variance.

Usage:
    # Single file
    python scripts/bootstrap_bpb.py eval_results/me340M-tied_bpe-128k_7Btok_seed42/bpb.parquet

    # All models under a directory (one row per bpb.parquet found)
    python scripts/bootstrap_bpb.py eval_results --glob "*/bpb.parquet" -o bpb_ci.parquet
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

LN2 = math.log(2)


def aggregate_bpb(loss_nats: np.ndarray, num_bytes: np.ndarray) -> float:
    """Byte-weighted corpus BPB."""
    return float(loss_nats.sum() / (num_bytes.sum() * LN2))


def bootstrap_bpb(
    df: pl.DataFrame,
    n_boot: int,
    seed: int,
    ci: float,
) -> dict:
    loss = df["loss_nats"].to_numpy()
    nbytes = df["num_bytes"].to_numpy()
    n = len(loss)

    point = aggregate_bpb(loss, nbytes)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample docs with replacement
        boot[b] = aggregate_bpb(loss[idx], nbytes[idx])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "n_docs": n,
        "bpb": point,
        "bpb_std": float(boot.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


@app.command()
def main(
    path: Path = typer.Argument(..., help="A bpb.parquet file, or a directory to search with --glob"),
    glob: str = typer.Option("*/bpb.parquet", "--glob", help="Glob used when PATH is a directory"),
    n_boot: int = typer.Option(10000, "--n-boot", help="Number of bootstrap resamples"),
    ci: float = typer.Option(0.95, "--ci", help="Confidence level (e.g. 0.95)"),
    seed: int = typer.Option(42, "--seed"),
    output: Path = typer.Option(None, "--output", "-o", help="Optional parquet to write the summary table"),
) -> None:
    if path.is_dir():
        files = sorted(path.glob(glob))
        if not files:
            console.print(f"[red]No files matching {glob} under {path}[/red]")
            raise typer.Exit(1)
    else:
        files = [path]

    rows = []
    for f in files:
        df = pl.read_parquet(f)
        missing = {"loss_nats", "num_bytes"} - set(df.columns)
        if missing:
            console.print(f"[yellow]Skipping {f}: missing columns {missing}[/yellow]")
            continue
        # model name = parent dir of the parquet (falls back to file stem)
        model = f.parent.name if f.parent != path else f.stem
        stats = bootstrap_bpb(df, n_boot=n_boot, seed=seed, ci=ci)
        rows.append({"model": model, **stats})

    if not rows:
        console.print("[red]No valid bpb parquet files processed.[/red]")
        raise typer.Exit(1)

    result = pl.DataFrame(rows).sort("bpb")

    pct = int(round(ci * 100))
    table = Table(title=f"Corpus BPB with {pct}% bootstrap CI ({n_boot} resamples)")
    table.add_column("model", overflow="fold")
    table.add_column("n_docs", justify="right")
    table.add_column("BPB", justify="right")
    table.add_column(f"{pct}% CI", justify="right")
    table.add_column("±half-width", justify="right")
    for r in result.iter_rows(named=True):
        half = (r["ci_high"] - r["ci_low"]) / 2.0
        table.add_row(
            r["model"],
            f"{r['n_docs']:,}",
            f"{r['bpb']:.4f}",
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
            f"{half:.4f}",
        )
    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output)
        console.print(f"[green]Wrote summary to {output}[/green]")


if __name__ == "__main__":
    app()
