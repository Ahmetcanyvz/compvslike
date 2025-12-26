"""Data validation utilities for tokenized datasets."""

import sys
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, load_from_disk
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Validate tokenized datasets for training.")
console = Console()


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


def validate_dataset(
    data_path: Path,
    vocab_size: int | None = None,
    expected_columns: list[str] | None = None,
    min_sequence_length: int = 1,
    max_sequence_length: int | None = None,
) -> dict:
    """Validate a tokenized Arrow dataset.

    Args:
        data_path: Path to the Arrow dataset directory.
        vocab_size: Maximum valid token ID (exclusive). If provided, checks all tokens are < vocab_size.
        expected_columns: List of required column names. Defaults to ["input_ids"].
        min_sequence_length: Minimum allowed sequence length.
        max_sequence_length: Maximum allowed sequence length (optional).

    Returns:
        Dictionary with validation statistics.

    Raises:
        DatasetValidationError: If validation fails.
    """
    if expected_columns is None:
        expected_columns = ["input_ids"]

    errors = []
    stats = {}

    # Check path exists
    if not data_path.exists():
        raise DatasetValidationError(f"Dataset path does not exist: {data_path}")

    # Load dataset
    try:
        dataset: Dataset = load_from_disk(str(data_path))
    except Exception as e:
        raise DatasetValidationError(f"Failed to load dataset: {e}") from e

    stats["num_examples"] = len(dataset)
    stats["columns"] = dataset.column_names

    # Check required columns
    for col in expected_columns:
        if col not in dataset.column_names:
            errors.append(f"Missing required column: {col}")

    if errors:
        raise DatasetValidationError("\n".join(errors))

    # Analyze input_ids
    if "input_ids" in dataset.column_names:
        # Sample statistics (avoid loading entire dataset into memory)
        sample_size = min(1000, len(dataset))
        sample = dataset.select(range(sample_size))

        lengths = [len(x) for x in sample["input_ids"]]
        stats["min_length"] = min(lengths)
        stats["max_length"] = max(lengths)
        stats["avg_length"] = sum(lengths) / len(lengths)

        # Check sequence length constraints
        if min(lengths) < min_sequence_length:
            errors.append(f"Found sequences shorter than {min_sequence_length} tokens")

        if max_sequence_length is not None and max(lengths) > max_sequence_length:
            errors.append(f"Found sequences longer than {max_sequence_length} tokens")

        # Check vocabulary bounds
        if vocab_size is not None:
            for i, ids in enumerate(sample["input_ids"]):
                max_id = max(ids) if len(ids) > 0 else 0
                min_id = min(ids) if len(ids) > 0 else 0
                if max_id >= vocab_size:
                    errors.append(f"Token ID {max_id} >= vocab_size {vocab_size} in example {i}")
                    break
                if min_id < 0:
                    errors.append(f"Negative token ID {min_id} in example {i}")
                    break

            stats["max_token_id"] = max(max(ids) for ids in sample["input_ids"] if len(ids) > 0)
            stats["min_token_id"] = min(min(ids) for ids in sample["input_ids"] if len(ids) > 0)

    # Compute total tokens (approximation based on sample)
    if "input_ids" in dataset.column_names:
        total_tokens_sample = sum(len(x) for x in sample["input_ids"])
        stats["estimated_total_tokens"] = int(total_tokens_sample * len(dataset) / sample_size)

    if errors:
        raise DatasetValidationError("\n".join(errors))

    return stats


def print_validation_report(data_path: Path, stats: dict) -> None:
    """Print a formatted validation report."""
    table = Table(title=f"Dataset Validation: {data_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Path", str(data_path))
    table.add_row("Examples", f"{stats['num_examples']:,}")
    table.add_row("Columns", ", ".join(stats["columns"]))

    if "min_length" in stats:
        table.add_row("Min sequence length", str(stats["min_length"]))
        table.add_row("Max sequence length", str(stats["max_length"]))
        table.add_row("Avg sequence length", f"{stats['avg_length']:.1f}")

    if "min_token_id" in stats:
        table.add_row("Token ID range", f"{stats['min_token_id']} - {stats['max_token_id']}")

    if "estimated_total_tokens" in stats:
        tokens = stats["estimated_total_tokens"]
        if tokens >= 1e9:
            table.add_row("Estimated total tokens", f"{tokens / 1e9:.2f}B")
        elif tokens >= 1e6:
            table.add_row("Estimated total tokens", f"{tokens / 1e6:.2f}M")
        else:
            table.add_row("Estimated total tokens", f"{tokens:,}")

    console.print(table)


@app.command()
def validate(
    data_path: Path = typer.Argument(..., help="Path to the Arrow dataset directory"),
    vocab_size: Optional[int] = typer.Option(None, "--vocab-size", "-v", help="Expected vocabulary size"),
    min_length: int = typer.Option(1, "--min-length", help="Minimum sequence length"),
    max_length: Optional[int] = typer.Option(None, "--max-length", help="Maximum sequence length"),
) -> None:
    """Validate a tokenized dataset for training."""
    try:
        stats = validate_dataset(
            data_path=data_path,
            vocab_size=vocab_size,
            min_sequence_length=min_length,
            max_sequence_length=max_length,
        )
        print_validation_report(data_path, stats)
        console.print("\n[green]Validation passed![/green]")

    except DatasetValidationError as e:
        console.print(f"\n[red]Validation failed:[/red]\n{e}")
        sys.exit(1)


@app.command()
def compare(
    train_path: Path = typer.Argument(..., help="Path to training dataset"),
    val_path: Path = typer.Argument(..., help="Path to validation dataset"),
    test_path: Optional[Path] = typer.Option(None, "--test", help="Path to test dataset"),
) -> None:
    """Compare multiple dataset splits."""
    splits = [("train", train_path), ("val", val_path)]
    if test_path:
        splits.append(("test", test_path))

    table = Table(title="Dataset Comparison")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", style="green")
    table.add_column("Avg Length", style="green")
    table.add_column("Est. Tokens", style="green")

    for name, path in splits:
        try:
            stats = validate_dataset(path)
            tokens = stats.get("estimated_total_tokens", 0)
            token_str = f"{tokens / 1e6:.1f}M" if tokens >= 1e6 else f"{tokens:,}"
            table.add_row(
                name,
                f"{stats['num_examples']:,}",
                f"{stats.get('avg_length', 0):.1f}",
                token_str,
            )
        except DatasetValidationError as e:
            table.add_row(name, "[red]ERROR[/red]", str(e), "")

    console.print(table)


if __name__ == "__main__":
    app()
