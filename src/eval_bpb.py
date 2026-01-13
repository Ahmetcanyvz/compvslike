"""Bits-per-byte (BPB) evaluation for language models.

BPB normalizes perplexity by byte length, allowing fair comparison
across models trained with different tokenizers/vocabulary sizes.

Formula: BPB = total_loss_in_nats / (num_bytes * ln(2))
"""

import math
from pathlib import Path

import polars as pl
import torch
import typer
from datasets import Dataset, load_from_disk
from rich.console import Console
from rich.table import Table
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.model import load_model_from_checkpoint

app = typer.Typer(help="Compute bits-per-byte for trained models.")
console = Console()


def compute_sequence_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    window_size: int = 2048,
    step_size: int = 512,
) -> float:
    """Compute total cross-entropy loss for a sequence.

    Uses sliding window for sequences longer than window_size.

    Args:
        model: The language model.
        input_ids: Input token IDs of shape (seq_len,).
        window_size: Size of each context window.
        step_size: Step size between windows.

    Returns:
        Total loss in nats (sum of negative log-probs).
    """
    input_ids = input_ids.unsqueeze(0)  # Add batch dim
    seq_len = input_ids.shape[1]

    if seq_len <= 1:
        return 0.0

    # For short sequences, process in one go
    if seq_len <= window_size:
        labels = input_ids[:, 1:]
        logits = model(input_ids[:, :-1]).logits
        loss = cross_entropy(
            logits.permute(0, 2, 1),
            labels,
            reduction="sum",
        )
        return loss.item()

    # Sliding window for long sequences
    total_loss = 0.0
    prev_end = 0

    for start in range(0, seq_len - 1, step_size):
        end = min(start + window_size, seq_len - 1)

        input_chunk = input_ids[:, start:end]
        label_chunk = input_ids[:, start + 1 : end + 1]

        logits = model(input_chunk).logits

        # Compute per-position loss
        loss_chunk = cross_entropy(
            logits.permute(0, 2, 1),
            label_chunk,
            reduction="none",
        )

        # For overlapping windows, only count new positions
        if start != 0:
            s = min(end - prev_end, step_size)
            loss_chunk = loss_chunk[:, -s:]

        total_loss += loss_chunk.sum().item()
        prev_end = end

        if end == seq_len - 1:
            break

    return total_loss


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    tokenizer_path: Path = typer.Argument(..., help="Path to tokenizer"),
    data_path: Path = typer.Argument(..., help="Path to raw text dataset (with 'text' column)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output parquet file (optional)"),
    max_samples: int = typer.Option(None, "--max-samples", "-n", help="Max samples to evaluate"),
    window_size: int = typer.Option(2048, "--window-size", help="Context window size"),
    step_size: int = typer.Option(512, "--step-size", help="Sliding window step size"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda/cpu)"),
) -> None:
    """Compute bits-per-byte for a dataset.

    Requires a dataset with a 'text' column containing raw text.
    """
    console.print(f"[green]Loading model from {checkpoint}[/green]")

    # Load model
    model = load_model_from_checkpoint(checkpoint)
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    console.print(f"[blue]Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters[/blue]")

    # Load tokenizer
    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    console.print(f"[blue]Tokenizer vocab size: {len(tokenizer)}[/blue]")

    # Load dataset
    console.print(f"[green]Loading data from {data_path}[/green]")
    dataset: Dataset = load_from_disk(str(data_path))

    if "text" not in dataset.column_names:
        console.print("[red]Error: Dataset must have a 'text' column[/red]")
        raise typer.Exit(1)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    console.print(f"[blue]Evaluating {len(dataset)} documents[/blue]")

    # Process documents
    results = []
    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0

    with torch.inference_mode():
        for example in tqdm(dataset, desc="Computing BPB"):
            text = example["text"]

            # Count bytes
            num_bytes = len(text.encode("utf-8"))
            if num_bytes == 0:
                continue

            # Tokenize
            input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
            input_ids = input_ids.squeeze(0).to(device)
            num_tokens = len(input_ids)

            if num_tokens <= 1:
                continue

            # Compute loss
            loss = compute_sequence_loss(model, input_ids, window_size, step_size)

            # Compute BPB for this document
            bpb = loss / (num_bytes * math.log(2))

            results.append({
                "uid": example.get("uid", len(results)),
                "num_bytes": num_bytes,
                "num_tokens": num_tokens,
                "loss_nats": loss,
                "bpb": bpb,
            })

            total_loss += loss
            total_bytes += num_bytes
            total_tokens += num_tokens

    # Compute aggregate BPB
    aggregate_bpb = total_loss / (total_bytes * math.log(2))

    # Create results dataframe
    df = pl.DataFrame(results)

    # Print statistics
    console.print("\n")
    table = Table(title="Bits-per-Byte Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total documents", f"{len(results):,}")
    table.add_row("Total bytes", f"{total_bytes:,}")
    table.add_row("Total tokens", f"{total_tokens:,}")
    table.add_row("Bytes per token", f"{total_bytes / total_tokens:.2f}")
    table.add_row("", "")
    table.add_row("Aggregate BPB", f"{aggregate_bpb:.4f}")
    table.add_row("Mean doc BPB", f"{df['bpb'].mean():.4f}")
    table.add_row("Std doc BPB", f"{df['bpb'].std():.4f}")
    table.add_row("Min doc BPB", f"{df['bpb'].min():.4f}")
    table.add_row("Max doc BPB", f"{df['bpb'].max():.4f}")

    console.print(table)

    # Compute perplexity for reference
    avg_loss_per_token = total_loss / total_tokens
    perplexity = math.exp(avg_loss_per_token)
    console.print(f"\n[dim]Reference: Perplexity = {perplexity:.2f}[/dim]")

    # Save results if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        console.print(f"\n[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    app()
