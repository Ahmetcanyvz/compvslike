"""Data preparation utilities for tokenizing text datasets."""

from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, DatasetDict, load_dataset
from rich.console import Console
from rich.progress import Progress
from transformers import AutoTokenizer

app = typer.Typer(help="Prepare data for language model training.")
console = Console()


@app.command()
def tokenize(
    dataset_name: str = typer.Argument(..., help="HuggingFace dataset name or local path"),
    tokenizer_path: str = typer.Argument(..., help="Path to tokenizer"),
    output_dir: Path = typer.Argument(..., help="Output directory for tokenized data"),
    text_column: str = typer.Option("text", "--text-column", "-t", help="Column containing text"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="Dataset split to use"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="Max samples to process"),
    num_proc: int = typer.Option(8, "--num-proc", help="Number of processes"),
    train_ratio: float = typer.Option(0.95, "--train-ratio", help="Train/val split ratio"),
) -> None:
    """Tokenize a text dataset for training.

    Examples:
        # Tokenize MiniPile
        python -m src.prepare_data tokenize JeanKaddworacle/minipile ./tokenizer ./data

        # Tokenize local jsonl files
        python -m src.prepare_data tokenize ./raw_data ./tokenizer ./data --text-column content

        # Tokenize with custom split
        python -m src.prepare_data tokenize openwebtext ./tokenizer ./data --split train[:10%]
    """
    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Ensure tokenizer has required tokens
    if tokenizer.eos_token_id is None:
        console.print("[yellow]Warning: Tokenizer has no EOS token, using token 0[/yellow]")

    console.print(f"[green]Loading dataset: {dataset_name}[/green]")

    # Load dataset
    if Path(dataset_name).exists():
        # Local files
        if Path(dataset_name).is_dir():
            files = list(Path(dataset_name).glob("*.jsonl")) + list(Path(dataset_name).glob("*.json"))
            dataset = load_dataset("json", data_files=[str(f) for f in files], split="train")
        else:
            dataset = load_dataset("json", data_files=dataset_name, split="train")
    else:
        # HuggingFace Hub
        if split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name)
            if isinstance(dataset, DatasetDict):
                # Combine all splits
                dataset = dataset["train"] if "train" in dataset else list(dataset.values())[0]

    console.print(f"[blue]Loaded {len(dataset)} samples[/blue]")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        console.print(f"[blue]Using {len(dataset)} samples[/blue]")

    # Check text column exists
    if text_column not in dataset.column_names:
        console.print(f"[red]Column '{text_column}' not found. Available: {dataset.column_names}[/red]")
        raise typer.Exit(1)

    # Tokenize
    console.print("[green]Tokenizing...[/green]")

    def tokenize_fn(examples):
        return {
            "input_ids": tokenizer(
                examples[text_column],
                add_special_tokens=False,  # We add EOS during packing
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
        }

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Add uid column
    tokenized = tokenized.map(
        lambda x, idx: {"uid": idx},
        with_indices=True,
        num_proc=num_proc,
        desc="Adding UIDs",
    )

    # Compute stats
    total_tokens = sum(len(x) for x in tokenized["input_ids"])
    avg_len = total_tokens / len(tokenized)

    console.print(f"[blue]Total tokens: {total_tokens:,}[/blue]")
    console.print(f"[blue]Average document length: {avg_len:.1f} tokens[/blue]")

    # Split into train/val
    console.print("[green]Splitting into train/val...[/green]")

    split_dataset = tokenized.train_test_split(
        test_size=1 - train_ratio,
        seed=42,
    )

    # Save
    output_dir = Path(output_dir)
    train_path = output_dir / "train"
    val_path = output_dir / "val"

    console.print(f"[green]Saving to {output_dir}[/green]")

    split_dataset["train"].save_to_disk(str(train_path))
    split_dataset["test"].save_to_disk(str(val_path))

    console.print(f"[green]Done![/green]")
    console.print(f"  Train: {len(split_dataset['train'])} documents -> {train_path}")
    console.print(f"  Val: {len(split_dataset['test'])} documents -> {val_path}")


@app.command()
def from_text_files(
    input_dir: Path = typer.Argument(..., help="Directory containing text files"),
    tokenizer_path: str = typer.Argument(..., help="Path to tokenizer"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    pattern: str = typer.Option("*.txt", "--pattern", help="File pattern"),
    train_ratio: float = typer.Option(0.95, "--train-ratio", help="Train/val split ratio"),
) -> None:
    """Tokenize text files from a directory.

    Each file becomes one document.
    """
    console.print(f"[green]Loading tokenizer from {tokenizer_path}[/green]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    files = list(Path(input_dir).glob(pattern))
    console.print(f"[blue]Found {len(files)} files[/blue]")

    if not files:
        console.print("[red]No files found![/red]")
        raise typer.Exit(1)

    # Read and tokenize files
    documents = []
    for i, file in enumerate(files):
        text = file.read_text()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        documents.append({"input_ids": tokens, "uid": i})

    dataset = Dataset.from_list(documents)

    # Split and save
    split_dataset = dataset.train_test_split(test_size=1 - train_ratio, seed=42)

    output_dir = Path(output_dir)
    split_dataset["train"].save_to_disk(str(output_dir / "train"))
    split_dataset["test"].save_to_disk(str(output_dir / "val"))

    total_tokens = sum(len(d["input_ids"]) for d in documents)
    console.print(f"[green]Done! Total tokens: {total_tokens:,}[/green]")


@app.command()
def download_minipile(
    tokenizer_path: str = typer.Argument(..., help="Path to tokenizer"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
) -> None:
    """Download and tokenize MiniPile dataset (~1.5B tokens).

    Good for training 57M-100M models.
    """
    console.print("[green]Downloading MiniPile...[/green]")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load MiniPile
    dataset = load_dataset("JeanKaddour/minipile", split="train")
    console.print(f"[blue]Loaded {len(dataset)} documents[/blue]")

    # Tokenize
    def tokenize_fn(examples):
        return {
            "input_ids": tokenizer(
                examples["text"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
        }

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    tokenized = tokenized.map(
        lambda x, idx: {"uid": idx},
        with_indices=True,
        num_proc=8,
    )

    # Use official splits
    val_dataset = load_dataset("JeanKaddour/minipile", split="validation")
    test_dataset = load_dataset("JeanKaddour/minipile", split="test")

    val_tokenized = val_dataset.map(
        tokenize_fn, batched=True, num_proc=8, remove_columns=val_dataset.column_names
    ).map(lambda x, idx: {"uid": idx}, with_indices=True, num_proc=8)

    test_tokenized = test_dataset.map(
        tokenize_fn, batched=True, num_proc=8, remove_columns=test_dataset.column_names
    ).map(lambda x, idx: {"uid": idx}, with_indices=True, num_proc=8)

    # Save
    output_dir = Path(output_dir)
    tokenized.save_to_disk(str(output_dir / "train"))
    val_tokenized.save_to_disk(str(output_dir / "val"))
    test_tokenized.save_to_disk(str(output_dir / "test"))

    total_tokens = sum(len(x) for x in tokenized["input_ids"])
    console.print(f"[green]Done! Total training tokens: {total_tokens:,} (~{total_tokens/1e9:.2f}B)[/green]")


@app.command()
def stats(
    data_path: Path = typer.Argument(..., help="Path to tokenized dataset"),
) -> None:
    """Show statistics for a tokenized dataset."""
    from datasets import load_from_disk

    dataset = load_from_disk(str(data_path))

    lengths = [len(x) for x in dataset["input_ids"]]
    total_tokens = sum(lengths)

    console.print(f"Documents: {len(dataset):,}")
    console.print(f"Total tokens: {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    console.print(f"Avg length: {total_tokens/len(dataset):.1f}")
    console.print(f"Min length: {min(lengths)}")
    console.print(f"Max length: {max(lengths)}")


if __name__ == "__main__":
    app()
