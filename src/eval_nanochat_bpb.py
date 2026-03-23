"""BPB evaluation for nanochat GPT models."""

import json
import math
import sys
from pathlib import Path

import torch
import typer
from datasets import load_from_disk
from rich.console import Console
from rich.table import Table
from torch.nn.functional import cross_entropy
from tqdm.auto import tqdm

NANOCHAT_DIR = Path(__file__).parent.parent / "nanochat"
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

from nanochat.gpt import GPT, GPTConfig

app = typer.Typer()
console = Console()


def load_nanochat_model(checkpoint_path: Path, meta_path: Path, device: str = "cuda") -> GPT:
    """Load a nanochat GPT model from checkpoint."""
    with open(meta_path) as f:
        meta = json.load(f)
    config = GPTConfig(**meta["model_config"])
    model = GPT(config)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model


def load_tokenizer(tokenizer_dir: Path):
    """Load tokenizer — try HuggingFace json first, fall back to tiktoken pkl."""
    json_path = tokenizer_dir / "tokenizer.json"
    pkl_path = tokenizer_dir / "tokenizer.pkl"

    if json_path.exists():
        try:
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast(tokenizer_file=str(json_path))
            return tok, "hf"
        except Exception:
            pass

    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            tok = pickle.load(f)
        return tok, "tiktoken"

    raise FileNotFoundError(f"No tokenizer found in {tokenizer_dir}")


def encode(tokenizer, tok_type: str, text: str) -> list[int]:
    if tok_type == "hf":
        return tokenizer.encode(text, add_special_tokens=False)
    else:
        return tokenizer.encode(text)


def compute_sequence_loss(model: GPT, input_ids: torch.Tensor, window_size: int = 2048, step_size: int = 512) -> float:
    """Compute total cross-entropy loss for a sequence with sliding window."""
    input_ids = input_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]

    if seq_len <= 1:
        return 0.0

    if seq_len <= window_size:
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        return loss.item()

    total_loss = 0.0
    prev_end = 0

    for start in range(0, seq_len - 1, step_size):
        end = min(start + window_size, seq_len - 1)
        x = input_ids[:, start:end]
        y = input_ids[:, start + 1:end + 1]
        logits = model(x)
        loss_chunk = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        loss_chunk = loss_chunk.view(1, -1)

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
    checkpoint_dir: Path = typer.Argument(..., help="Directory containing model_*.pt and meta_*.json"),
    tokenizer_dir: Path = typer.Argument(..., help="Directory containing tokenizer"),
    data_path: Path = typer.Argument(..., help="Path to raw text dataset (with 'text' column)"),
    output: Path = typer.Option(None, "--output", "-o", help="Output parquet file"),
    max_samples: int = typer.Option(None, "--max-samples", "-n"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Compute bits-per-byte for a nanochat model."""
    import polars as pl

    # Find checkpoint and meta files
    model_files = sorted(checkpoint_dir.glob("model_*.pt"))
    meta_files = sorted(checkpoint_dir.glob("meta_*.json"))
    if not model_files or not meta_files:
        console.print(f"[red]No model/meta files found in {checkpoint_dir}[/red]")
        raise typer.Exit(1)

    model_path = model_files[-1]
    meta_path = meta_files[-1]
    console.print(f"[green]Loading model from {model_path}[/green]")

    model = load_nanochat_model(model_path, meta_path, device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"[blue]Model: {num_params:.1f}M parameters[/blue]")

    tokenizer, tok_type = load_tokenizer(tokenizer_dir)
    console.print(f"[green]Loaded {tok_type} tokenizer from {tokenizer_dir}[/green]")

    console.print(f"[green]Loading data from {data_path}[/green]")
    dataset = load_from_disk(str(data_path))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    console.print(f"[blue]Evaluating {len(dataset)} documents[/blue]")

    results = []
    total_loss = 0.0
    total_bytes = 0
    total_tokens = 0

    with torch.inference_mode():
        for example in tqdm(dataset, desc="Computing BPB"):
            text = example["text"]
            num_bytes = len(text.encode("utf-8"))
            if num_bytes == 0:
                continue

            ids = encode(tokenizer, tok_type, text)
            input_ids = torch.tensor(ids, device=device)
            num_tokens = len(input_ids)

            if num_tokens <= 1:
                continue

            loss = compute_sequence_loss(model, input_ids)
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

    aggregate_bpb = total_loss / (total_bytes * math.log(2))
    perplexity = math.exp(total_loss / total_tokens)

    df = pl.DataFrame(results)

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
    table.add_row("Perplexity", f"{perplexity:.2f}")
    console.print(table)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        console.print(f"[green]Saved to {output}[/green]")


if __name__ == "__main__":
    app()
