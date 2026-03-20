"""End-to-end data preparation: download raw data + tokenize for training.

Downloads FineWeb-Edu raw text (if not already present), then tokenizes it
with the specified tokenizer. Produces train/val/test splits ready for training.

Usage:
    # Full pipeline: download + tokenize
    uv run python scripts/prepare_all.py \
        --tokenizer /path/to/tokenizer/bpe-32k \
        --output-dir data/fineweb-edu-bpe-32k \
        --target-tokens 2_000_000_000

    # Skip download if raw data already exists
    uv run python scripts/prepare_all.py \
        --tokenizer /path/to/tokenizer/bpe-32k \
        --output-dir data/fineweb-edu-bpe-32k \
        --raw-data-dir data/fineweb-edu-raw

    # Tokenize all tokenizers at once
    uv run python scripts/prepare_all.py \
        --tokenizer /path/to/tokenizers/bpe-8k /path/to/tokenizers/bpe-32k \
        --output-dir data \
        --raw-data-dir data/fineweb-edu-raw
"""

import random
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, load_dataset, load_from_disk
from rich.console import Console
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer(help="End-to-end data preparation for LM training.")
console = Console()


def download_raw_data(
    output_dir: Path,
    target_tokens: int,
    seed: int,
    min_tokens: int,
    train_ratio: float,
    val_ratio: float,
) -> Path:
    """Download FineWeb-Edu and create train/val/test splits of raw text."""
    if (output_dir / "train").exists() and (output_dir / "test").exists():
        console.print(f"[yellow]Raw data already exists at {output_dir}, skipping download.[/yellow]")
        return output_dir

    console.print(f"[green]Downloading FineWeb-Edu (~{target_tokens / 1e9:.0f}B tokens)...[/green]")

    estimator = AutoTokenizer.from_pretrained("gpt2")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    documents = []
    total_tokens = 0

    pbar = tqdm(total=target_tokens, unit="tok", desc="Collecting data")

    for example in dataset:
        text = example["text"]
        est_tokens = len(estimator.encode(text, add_special_tokens=False))

        if est_tokens < min_tokens:
            continue

        documents.append({
            "text": text,
            "uid": len(documents),
        })

        total_tokens += est_tokens
        pbar.update(est_tokens)

        if total_tokens >= target_tokens:
            break

        if len(documents) % 10000 == 0:
            pbar.set_postfix({"docs": len(documents)})

    pbar.close()
    console.print(f"Collected {len(documents):,} documents ({total_tokens / 1e9:.2f}B tokens)")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(documents)

    n = len(documents)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    splits = {
        "train": documents[:train_end],
        "val": documents[train_end:val_end],
        "test": documents[val_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, docs in splits.items():
        ds = Dataset.from_list(docs)
        ds.save_to_disk(str(output_dir / split_name))
        console.print(f"  {split_name}: {len(docs):,} documents")

    console.print(f"[green]Raw data saved to {output_dir}[/green]")
    return output_dir


def tokenize_split(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_proc: int,
) -> Dataset:
    """Tokenize a single split."""
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        return {
            "input_ids": tokens["input_ids"],
            "uid": examples["uid"],
        }

    return raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )


def tokenize_raw_data(
    raw_data_dir: Path,
    tokenizer_path: str,
    output_dir: Path,
    num_proc: int,
) -> None:
    """Tokenize raw text data with a specific tokenizer."""
    if (output_dir / "train").exists():
        console.print(f"[yellow]Tokenized data already exists at {output_dir}, skipping.[/yellow]")
        return

    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    console.print(f"  Vocab size: {len(tokenizer):,}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_path = raw_data_dir / split_name
        if not split_path.exists():
            console.print(f"[yellow]  {split_name} split not found, skipping.[/yellow]")
            continue

        console.print(f"[green]  Tokenizing {split_name}...[/green]")
        raw_ds = load_from_disk(str(split_path))
        tok_ds = tokenize_split(raw_ds, tokenizer, num_proc)

        # Print stats
        total_tokens = sum(len(x) for x in tok_ds["input_ids"])
        console.print(f"    {len(tok_ds):,} docs, {total_tokens:,} tokens ({total_tokens / 1e9:.2f}B)")

        tok_ds.save_to_disk(str(output_dir / split_name))

    console.print(f"[green]Tokenized data saved to {output_dir}[/green]")


@app.command()
def main(
    tokenizer: list[str] = typer.Option(..., "--tokenizer", "-t", help="Path(s) to tokenizer(s)"),
    output_dir: Path = typer.Option("data", "--output-dir", "-o", help="Base output directory"),
    raw_data_dir: Optional[Path] = typer.Option(None, "--raw-data-dir", help="Existing raw data directory (skip download)"),
    target_tokens: int = typer.Option(2_000_000_000, "--target-tokens", help="Target token count for download"),
    seed: int = typer.Option(42, "--seed", help="Random seed for splitting"),
    train_ratio: float = typer.Option(0.95, "--train-ratio", help="Train split ratio"),
    val_ratio: float = typer.Option(0.025, "--val-ratio", help="Val split ratio"),
    min_tokens: int = typer.Option(50, "--min-tokens", help="Minimum tokens per document"),
    num_proc: int = typer.Option(8, "--num-proc", help="Number of processes for tokenization"),
) -> None:
    """Download FineWeb-Edu and tokenize for training.

    If --raw-data-dir is provided, skips download and uses existing raw data.
    Multiple tokenizers can be specified to tokenize in one go.
    """
    # Step 1: Get raw data
    if raw_data_dir and raw_data_dir.exists():
        console.print(f"[green]Using existing raw data from {raw_data_dir}[/green]")
    else:
        raw_data_dir = output_dir / "fineweb-edu-raw"
        download_raw_data(
            output_dir=raw_data_dir,
            target_tokens=target_tokens,
            seed=seed,
            min_tokens=min_tokens,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    # Step 2: Tokenize with each tokenizer
    for tok_path in tokenizer:
        tok_name = Path(tok_path).name
        tok_output = output_dir / f"fineweb-edu-{tok_name}"
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Tokenizer: {tok_name}[/bold]")
        console.print(f"{'=' * 60}")
        tokenize_raw_data(raw_data_dir, tok_path, tok_output, num_proc)

    console.print(f"\n[green bold]All done![/green bold]")


if __name__ == "__main__":
    app()
