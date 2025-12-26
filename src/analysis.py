"""Regression discontinuity analysis for tokenisation bias estimation."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import statsmodels.api as sm
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Regression discontinuity analysis for tokenisation bias.")
console = Console()


def load_merge_info(tokenizer_path: Path) -> pl.DataFrame:
    """Load tokenizer merge information.

    Expects a tokenizer with merge_counts.json file containing merge order.
    """
    merge_counts_path = tokenizer_path / "merge_counts.json"

    if not merge_counts_path.exists():
        raise FileNotFoundError(f"merge_counts.json not found in {tokenizer_path}")

    with open(merge_counts_path) as f:
        merge_counts = json.load(f)

    # Convert to dataframe
    records = []
    for merge_str, info in merge_counts.items():
        records.append(
            {
                "merge": merge_str,
                "token_id": info.get("token_id"),
                "merge_order": info.get("merge_order"),
                "count": info.get("count", 0),
            }
        )

    return pl.DataFrame(records)


def load_eval_results(eval_path: Path) -> pl.DataFrame:
    """Load evaluation results from parquet file."""
    return pl.read_parquet(eval_path)


def compute_token_statistics(eval_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-token statistics from evaluation results.

    Aggregates log-probabilities across all occurrences of each token.
    """
    # Explode the lists to get individual token-logprob pairs
    exploded = eval_df.select(
        pl.col("uid"),
        pl.col("token_ids").explode().alias("token_id"),
        pl.col("token_logprobs").explode().alias("logprob"),
    )

    # Aggregate by token_id
    token_stats = exploded.group_by("token_id").agg(
        pl.col("logprob").mean().alias("mean_logprob"),
        pl.col("logprob").std().alias("std_logprob"),
        pl.col("logprob").count().alias("count"),
        pl.col("logprob").sum().alias("total_logprob"),
    )

    return token_stats


def prepare_rd_data(
    token_stats: pl.DataFrame,
    merge_info: pl.DataFrame,
    cutoff_vocab_size: int,
) -> pl.DataFrame:
    """Prepare data for regression discontinuity analysis.

    Args:
        token_stats: Per-token statistics from evaluation.
        merge_info: Tokenizer merge information.
        cutoff_vocab_size: The vocabulary size cutoff for treatment assignment.

    Returns:
        DataFrame with running variable (merge_order) and treatment indicator.
    """
    # Join token stats with merge info
    merged = token_stats.join(merge_info, on="token_id", how="inner")

    # Add treatment indicator: 1 if token is in vocabulary (merge_order < cutoff)
    merged = merged.with_columns(
        (pl.col("merge_order") < cutoff_vocab_size).cast(pl.Int32).alias("treatment"),
        (pl.col("merge_order") - cutoff_vocab_size).alias("running_var"),
    )

    return merged


def run_rd_regression(
    data: pl.DataFrame,
    bandwidth: int = 1000,
    kernel: str = "triangular",
) -> dict:
    """Run regression discontinuity regression.

    Uses local linear regression around the cutoff.

    Args:
        data: Prepared RD data with running_var and treatment columns.
        bandwidth: Bandwidth around cutoff for local regression.
        kernel: Kernel type ('triangular' or 'uniform').

    Returns:
        Dictionary with regression results.
    """
    # Filter to bandwidth around cutoff
    subset = data.filter(pl.col("running_var").abs() <= bandwidth)

    if len(subset) < 10:
        raise ValueError(f"Too few observations within bandwidth: {len(subset)}")

    # Convert to numpy
    y = subset["mean_logprob"].to_numpy()
    running_var = subset["running_var"].to_numpy()
    treatment = subset["treatment"].to_numpy()

    # Compute kernel weights
    if kernel == "triangular":
        weights = 1 - np.abs(running_var) / bandwidth
    else:  # uniform
        weights = np.ones_like(running_var)

    # Build design matrix: intercept, treatment, running_var, treatment * running_var
    X = np.column_stack(
        [
            np.ones_like(running_var),
            treatment,
            running_var,
            treatment * running_var,
        ]
    )

    # Weighted least squares
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()

    # Extract treatment effect (coefficient on treatment indicator)
    treatment_effect = results.params[1]
    treatment_se = results.bse[1]
    treatment_pvalue = results.pvalues[1]

    return {
        "treatment_effect": treatment_effect,
        "standard_error": treatment_se,
        "p_value": treatment_pvalue,
        "t_statistic": results.tvalues[1],
        "ci_lower": treatment_effect - 1.96 * treatment_se,
        "ci_upper": treatment_effect + 1.96 * treatment_se,
        "n_observations": len(subset),
        "bandwidth": bandwidth,
        "r_squared": results.rsquared,
    }


def print_rd_results(results: dict) -> None:
    """Print regression discontinuity results in a formatted table."""
    table = Table(title="Regression Discontinuity Results")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Treatment Effect", f"{results['treatment_effect']:.6f}")
    table.add_row("Standard Error", f"{results['standard_error']:.6f}")
    table.add_row("t-statistic", f"{results['t_statistic']:.4f}")
    table.add_row("p-value", f"{results['p_value']:.4f}")
    table.add_row("95% CI", f"[{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    table.add_row("Observations", str(results["n_observations"]))
    table.add_row("Bandwidth", str(results["bandwidth"]))
    table.add_row("R-squared", f"{results['r_squared']:.4f}")

    console.print(table)

    # Interpretation
    if results["p_value"] < 0.05:
        console.print(
            f"\n[green]The treatment effect is statistically significant (p < 0.05).[/green]\n"
            f"Being in vocabulary is associated with a {results['treatment_effect']:.4f} change in mean log-probability."
        )
    else:
        console.print(f"\n[yellow]The treatment effect is not statistically significant (p = {results['p_value']:.4f}).[/yellow]")


@app.command()
def analyze(
    eval_path: Path = typer.Argument(..., help="Path to evaluation parquet file"),
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer directory"),
    cutoff_vocab_size: int = typer.Argument(..., help="Vocabulary size cutoff for treatment"),
    bandwidth: int = typer.Option(1000, "--bandwidth", "-b", help="Bandwidth around cutoff"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file for results"),
) -> None:
    """Run regression discontinuity analysis."""
    console.print("[green]Loading data...[/green]")

    # Load data
    eval_df = load_eval_results(eval_path)
    merge_info = load_merge_info(tokenizer_path)

    console.print(f"[blue]Loaded {len(eval_df)} documents[/blue]")
    console.print(f"[blue]Loaded {len(merge_info)} merge operations[/blue]")

    # Compute token statistics
    console.print("[green]Computing token statistics...[/green]")
    token_stats = compute_token_statistics(eval_df)
    console.print(f"[blue]Computed stats for {len(token_stats)} unique tokens[/blue]")

    # Prepare RD data
    console.print("[green]Preparing regression discontinuity data...[/green]")
    rd_data = prepare_rd_data(token_stats, merge_info, cutoff_vocab_size)
    console.print(f"[blue]RD data: {len(rd_data)} tokens with merge info[/blue]")

    # Run regression
    console.print("[green]Running regression...[/green]")
    results = run_rd_regression(rd_data, bandwidth=bandwidth)

    # Print results
    print_rd_results(results)

    # Save if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def token_stats(
    eval_path: Path = typer.Argument(..., help="Path to evaluation parquet file"),
    output: Path = typer.Argument(..., help="Output parquet file for token statistics"),
) -> None:
    """Compute and save per-token statistics."""
    console.print("[green]Loading evaluation results...[/green]")
    eval_df = load_eval_results(eval_path)

    console.print("[green]Computing token statistics...[/green]")
    stats = compute_token_statistics(eval_df)

    output.parent.mkdir(parents=True, exist_ok=True)
    stats.write_parquet(output)

    console.print(f"[green]Saved statistics for {len(stats)} tokens to {output}[/green]")


@app.command()
def compare_models(
    eval_paths: list[Path] = typer.Argument(..., help="Paths to evaluation parquet files"),
    labels: Optional[str] = typer.Option(None, "--labels", help="Comma-separated labels for models"),
    tokenizer_path: Path = typer.Option(..., "--tokenizer", help="Path to tokenizer"),
    cutoff_vocab_size: int = typer.Option(..., "--cutoff", help="Vocabulary size cutoff"),
    bandwidth: int = typer.Option(1000, "--bandwidth", help="Bandwidth around cutoff"),
) -> None:
    """Compare regression discontinuity results across multiple models."""
    model_labels = labels.split(",") if labels else [f"Model {i}" for i in range(len(eval_paths))]

    if len(model_labels) != len(eval_paths):
        console.print("[red]Number of labels must match number of eval paths[/red]")
        raise typer.Exit(1)

    merge_info = load_merge_info(tokenizer_path)

    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Effect", style="green")
    table.add_column("SE", style="green")
    table.add_column("p-value", style="green")
    table.add_column("N", style="green")

    for label, eval_path in zip(model_labels, eval_paths):
        eval_df = load_eval_results(eval_path)
        token_stats = compute_token_statistics(eval_df)
        rd_data = prepare_rd_data(token_stats, merge_info, cutoff_vocab_size)

        try:
            results = run_rd_regression(rd_data, bandwidth=bandwidth)
            table.add_row(
                label,
                f"{results['treatment_effect']:.6f}",
                f"{results['standard_error']:.6f}",
                f"{results['p_value']:.4f}",
                str(results["n_observations"]),
            )
        except Exception as e:
            table.add_row(label, "[red]Error[/red]", str(e), "", "")

    console.print(table)


if __name__ == "__main__":
    app()
