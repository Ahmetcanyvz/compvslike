"""Evaluation script for collecting log-probabilities."""

from collections.abc import Generator
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
import typer
from datasets import Dataset, load_from_disk
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm

from src.model import load_model_from_checkpoint

app = typer.Typer(help="Collect log-probabilities from trained models.")
console = Console()


def batch_by_tokens(
    dataset: Dataset,
    max_tokens_per_batch: int,
) -> Generator[list[dict], None, None]:
    """Yield batches of documents with total tokens <= max_tokens_per_batch.

    Documents are assumed to be sorted by length (longest first) for efficiency.
    Uses greedy bin packing considering padding overhead.
    """
    current_batch = []
    current_max_length = 0

    for example in dataset:
        num_tokens = len(example["input_ids"])
        new_max_length = max(current_max_length, num_tokens)
        new_total_tokens = new_max_length * (len(current_batch) + 1)

        if new_total_tokens > max_tokens_per_batch and current_batch:
            yield current_batch
            current_batch = [example]
            current_max_length = num_tokens
        else:
            current_batch.append(example)
            current_max_length = new_max_length

    if current_batch:
        yield current_batch


def collate_with_left_padding(
    batch: list[dict],
    pad_value: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate batch with left padding (for autoregressive models)."""
    input_ids = [example["input_ids"] for example in batch]
    uids = [example.get("uid", i) for i, example in enumerate(batch)]

    max_length = max(len(ids) for ids in input_ids)

    # Left-pad sequences
    padded = np.vstack(
        [np.pad(ids, (max_length - len(ids), 0), mode="constant", constant_values=pad_value) for ids in input_ids]
    )

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "uid": uids,
    }


def compute_logprobs_sliding_window(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pad_value: int = 0,
    window_size: int = 2048,
    step_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probabilities using sliding window for long sequences.

    Args:
        model: The language model.
        input_ids: Input token IDs of shape (batch_size, seq_len).
        pad_value: Token ID used for padding (ignored in loss).
        window_size: Size of each context window.
        step_size: Step size between windows.

    Returns:
        Tuple of (log_probs, token_ids) for each position.
    """
    seq_len = input_ids.shape[1]

    # For short sequences, process in one go
    if seq_len <= window_size:
        labels = input_ids[:, 1:]
        logits = model(input_ids[:, :-1]).logits
        log_probs = cross_entropy(
            logits.permute(0, 2, 1),
            labels,
            reduction="none",
            ignore_index=pad_value,
        ).neg()
        return log_probs, labels

    # Sliding window for long sequences
    all_logprobs = []
    all_tokens = []
    prev_end = 0

    for start in range(0, seq_len - 1, step_size):
        end = min(start + window_size, seq_len - 1)

        input_chunk = input_ids[:, start:end]
        label_chunk = input_ids[:, start + 1 : end + 1]

        logits = model(input_chunk).logits
        logprob_chunk = cross_entropy(
            logits.permute(0, 2, 1),
            label_chunk,
            reduction="none",
            ignore_index=pad_value,
        ).neg()

        # For overlapping windows, only keep the new tokens
        if start != 0:
            s = min(end - prev_end, step_size)
            logprob_chunk = logprob_chunk[:, -s:]
            label_chunk = label_chunk[:, -s:]

        all_logprobs.append(logprob_chunk.cpu())
        all_tokens.append(label_chunk.cpu())

        prev_end = end
        if end == seq_len - 1:
            break

    log_probs = torch.cat(all_logprobs, dim=1)
    token_ids = torch.cat(all_tokens, dim=1)

    return log_probs, token_ids


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    data_path: Path = typer.Argument(..., help="Path to evaluation dataset"),
    output: Path = typer.Argument(..., help="Output parquet file path"),
    max_tokens_per_batch: int = typer.Option(20000, "--batch-tokens", help="Max tokens per batch"),
    window_size: int = typer.Option(2048, "--window-size", help="Context window size"),
    step_size: int = typer.Option(512, "--step-size", help="Sliding window step size"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
    precision: str = typer.Option("bf16", "--precision", help="Precision (fp32/fp16/bf16)"),
) -> None:
    """Collect log-probabilities for each token in the dataset."""
    console.print(f"[green]Loading model from {checkpoint}[/green]")

    # Load model
    model = load_model_from_checkpoint(checkpoint)
    model = model.to(device)
    model.eval()

    if precision == "bf16":
        model = model.to(torch.bfloat16)
    elif precision == "fp16":
        model = model.to(torch.float16)

    console.print(f"[blue]Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters[/blue]")

    # Load dataset
    console.print(f"[green]Loading data from {data_path}[/green]")
    dataset: Dataset = load_from_disk(str(data_path))

    # Add length column and sort by length (longest first for efficient batching)
    dataset = dataset.map(
        lambda x: {"length": len(x["input_ids"])},
        load_from_cache_file=False,
    ).sort("length", reverse=True)

    console.print(f"[blue]Loaded {len(dataset)} documents[/blue]")

    # Process batches
    results = []
    batch_generator = batch_by_tokens(dataset, max_tokens_per_batch)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(dataset))

        for batch_data in tqdm(batch_generator, desc="Evaluating"):
            batch = collate_with_left_padding(batch_data, pad_value=0)
            input_ids = batch["input_ids"].to(device)
            uids = batch["uid"]

            # Compute log-probs
            with torch.inference_mode():
                log_probs, token_ids = compute_logprobs_sliding_window(
                    model,
                    input_ids,
                    pad_value=0,
                    window_size=window_size,
                    step_size=step_size,
                )

            # Convert to lists and store
            for i, uid in enumerate(uids):
                # Find where padding ends (first non-zero for this example)
                seq_logprobs = log_probs[i].numpy()
                seq_tokens = token_ids[i].numpy()

                # Remove padding (where token_id == 0 at the start)
                # Note: This assumes 0 is the pad token
                non_pad_mask = seq_tokens != 0
                first_non_pad = np.argmax(non_pad_mask) if non_pad_mask.any() else 0

                results.append(
                    {
                        "uid": uid,
                        "token_ids": seq_tokens[first_non_pad:].tolist(),
                        "token_logprobs": seq_logprobs[first_non_pad:].tolist(),
                    }
                )

            progress.update(task, advance=len(batch_data))

    # Save results
    console.print(f"[green]Saving results to {output}[/green]")
    df = pl.DataFrame(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    console.print(f"[green]Done! Saved {len(results)} documents[/green]")


@app.command()
def aggregate(
    input_dir: Path = typer.Argument(..., help="Directory containing parquet files"),
    output: Path = typer.Argument(..., help="Output aggregated parquet file"),
    pattern: str = typer.Option("*.parquet", "--pattern", help="Glob pattern for input files"),
) -> None:
    """Aggregate multiple evaluation parquet files into one."""
    files = list(input_dir.glob(pattern))
    if not files:
        console.print(f"[red]No files found matching {pattern} in {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Found {len(files)} files to aggregate[/blue]")

    dfs = [pl.read_parquet(f) for f in files]
    combined = pl.concat(dfs)

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output)

    console.print(f"[green]Aggregated {len(combined)} documents to {output}[/green]")


if __name__ == "__main__":
    app()
