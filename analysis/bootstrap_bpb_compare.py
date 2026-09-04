"""Paired bootstrap for BPB *differences* between models on the same corpus.

Models evaluated on the same documents have strongly paired per-document
losses. Comparing their marginal 95% CIs for overlap is the WRONG test (it
ignores the shared per-document difficulty and is far too conservative).

Instead, resample document indices once per iteration and apply the SAME
indices to both models:

    Delta = BPB_A - BPB_B     (byte-weighted corpus BPB, each with own bytes)

The difference is significant at level (1 - ci) if the CI of Delta excludes 0.

Documents are aligned by `uid` via an inner join, so only documents present
in both files are compared.

Usage:
    # Two explicit models -> single Delta CI
    python scripts/bootstrap_bpb_compare.py \
        eval_results/me500M-tied_bottomupll-exact-128k_10Btok_seed42/bpb.parquet \
        eval_results/me500M-tied_bpe-128k_10Btok_seed42/bpb.parquet

    # A set of models -> every model vs a baseline (matched by substring)
    python scripts/bootstrap_bpb_compare.py eval_results \
        --glob "me500M-tied_*-128k_*/bpb.parquet" --baseline bpe -o cmp_500M.json

    # A set of models -> all unique pairs
    python scripts/bootstrap_bpb_compare.py eval_results \
        --glob "me500M-tied_*-128k_*/bpb.parquet" --all-pairs -o cmp_500M.json
"""

import math
from itertools import combinations
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


def _agg(loss: np.ndarray, nbytes: np.ndarray) -> float:
    return float(loss.sum() / (nbytes.sum() * LN2))


def _load(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    missing = {"uid", "loss_nats", "num_bytes"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.select("uid", "loss_nats", "num_bytes")


def paired_bootstrap(
    df_a: pl.DataFrame, df_b: pl.DataFrame, n_boot: int, seed: int, ci: float
) -> dict:
    j = df_a.join(df_b, on="uid", how="inner", suffix="_b")
    n = j.height
    if n == 0:
        raise ValueError("No overlapping uids between the two files.")

    loss_a = j["loss_nats"].to_numpy()
    bytes_a = j["num_bytes"].to_numpy()
    loss_b = j["loss_nats_b"].to_numpy()
    bytes_b = j["num_bytes_b"].to_numpy()

    delta = _agg(loss_a, bytes_a) - _agg(loss_b, bytes_b)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)  # shared indices -> paired
        boot[k] = _agg(loss_a[idx], bytes_a[idx]) - _agg(loss_b[idx], bytes_b[idx])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    # two-sided bootstrap p-value: mass on the opposite side of 0, doubled
    frac_gt = float((boot > 0).mean())
    p_boot = 2.0 * min(frac_gt, 1.0 - frac_gt)
    return {
        "n_docs": n,
        "delta_bpb": delta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": not (lo <= 0.0 <= hi),
        "p_boot": p_boot,
    }


def _model_name(path: Path, root: Optional[Path]) -> str:
    return path.parent.name if path.parent != root else path.stem


@app.command()
def main(
    path_a: Path = typer.Argument(..., help="A bpb.parquet file, or a directory (use --glob)"),
    path_b: Optional[Path] = typer.Argument(None, help="Second bpb.parquet (when PATH_A is a file)"),
    glob: str = typer.Option("*/bpb.parquet", "--glob", help="Glob when PATH_A is a directory"),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", help="Substring of the baseline model B; compare all others vs it"
    ),
    all_pairs: bool = typer.Option(False, "--all-pairs", help="Compare every unique pair"),
    n_boot: int = typer.Option(10000, "--n-boot"),
    ci: float = typer.Option(0.95, "--ci"),
    seed: int = typer.Option(42, "--seed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="json/jsonl/csv/parquet by extension"),
) -> None:
    pairs: list[tuple[str, Path, str, Path]] = []

    if path_a.is_dir():
        files = sorted(path_a.glob(glob))
        if len(files) < 2:
            console.print(f"[red]Need >=2 files; found {len(files)} matching {glob} under {path_a}[/red]")
            raise typer.Exit(1)
        named = [(_model_name(f, path_a), f) for f in files]

        if baseline is not None:
            base = [nf for nf in named if baseline in nf[0]]
            if len(base) != 1:
                console.print(f"[red]--baseline '{baseline}' matched {len(base)} models; must match exactly 1[/red]")
                raise typer.Exit(1)
            b_name, b_path = base[0]
            pairs = [(n, p, b_name, b_path) for (n, p) in named if p != b_path]
        else:
            # default to all-pairs when a directory is given without a baseline
            for (na, pa), (nb, pb) in combinations(named, 2):
                pairs.append((na, pa, nb, pb))
    else:
        if path_b is None:
            console.print("[red]PATH_A is a file; provide PATH_B (or pass a directory with --glob).[/red]")
            raise typer.Exit(1)
        pairs = [(_model_name(path_a, None), path_a, _model_name(path_b, None), path_b)]

    cache: dict[Path, pl.DataFrame] = {}

    def get(p: Path) -> pl.DataFrame:
        if p not in cache:
            cache[p] = _load(p)
        return cache[p]

    rows = []
    for a_name, a_path, b_name, b_path in pairs:
        stats = paired_bootstrap(get(a_path), get(b_path), n_boot=n_boot, seed=seed, ci=ci)
        rows.append({"model_a": a_name, "model_b": b_name, **stats})

    result = pl.DataFrame(rows).sort("delta_bpb")

    pct = int(round(ci * 100))
    table = Table(title=f"Paired BPB difference (A - B), {pct}% bootstrap CI ({n_boot} resamples)")
    table.add_column("A (lower=better)", overflow="fold")
    table.add_column("B", overflow="fold")
    table.add_column("ΔBPB", justify="right")
    table.add_column(f"{pct}% CI", justify="right")
    table.add_column("sig", justify="center")
    table.add_column("p", justify="right")
    for r in result.iter_rows(named=True):
        sig = "[green]✓[/green]" if r["significant"] else "[dim]·[/dim]"
        table.add_row(
            r["model_a"],
            r["model_b"],
            f"{r['delta_bpb']:+.5f}",
            f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]",
            sig,
            f"{r['p_boot']:.3f}",
        )
    console.print(table)
    console.print("[dim]ΔBPB<0 means A has lower (better) BPB than B. sig ✓ = CI excludes 0.[/dim]")

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
