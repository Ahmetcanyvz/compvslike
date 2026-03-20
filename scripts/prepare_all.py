"""End-to-end data preparation: download raw data + tokenize for training.

Val/test splits are always derived from the first 2B tokens (seed=42 shuffle),
matching the original notebook 01 procedure. If more training tokens are needed,
additional documents are streamed and appended only to the train split.

Usage:
    # Default 2B tokens (matches original setup)
    uv run python scripts/prepare_all.py \
        -t /path/to/tokenizers/bpe-32k \
        -o data

    # Scale up to 20B training tokens (same val/test)
    uv run python scripts/prepare_all.py \
        -t /path/to/tokenizers/bpe-32k \
        -o data \
        --target-tokens 20_000_000_000

    # Skip download if raw data already exists
    uv run python scripts/prepare_all.py \
        -t /path/to/tokenizers/bpe-32k \
        -o data \
        --raw-data-dir data/fineweb-edu-raw

    # Tokenize with multiple tokenizers at once
    uv run python scripts/prepare_all.py \
        -t /path/to/tokenizers/bpe-8k \
        -t /path/to/tokenizers/bpe-32k \
        -t /path/to/tokenizers/compmax-8k \
        -o data
"""

import random
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from rich.console import Console
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer(help="End-to-end data preparation for LM training.")
console = Console()

# These match the original notebook 01_prepare_fineweb.ipynb exactly
BASE_TOKENS = 2_000_000_000
BASE_SEED = 42
BASE_TRAIN_RATIO = 0.95
BASE_VAL_RATIO = 0.025
BASE_MIN_TOKENS = 50


def stream_documents(
    estimator: AutoTokenizer,
    target_tokens: int,
    min_tokens: int,
    skip_docs: int = 0,
    uid_offset: int = 0,
    desc: str = "Collecting data",
) -> tuple[list[dict], int, int]:
    """Stream documents from FineWeb-Edu until reaching target token count.

    Returns:
        (documents, total_tokens, total_streamed) where total_streamed includes skipped docs.
    """
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    documents = []
    total_tokens = 0
    streamed = 0

    if skip_docs > 0:
        pbar_skip = tqdm(total=skip_docs, unit="doc", desc="Skipping existing docs")

    pbar = tqdm(total=target_tokens, unit="tok", desc=desc)

    for example in dataset:
        if streamed < skip_docs:
            streamed += 1
            if skip_docs > 0:
                pbar_skip.update(1)
                if streamed == skip_docs:
                    pbar_skip.close()
                    console.print(f"Done skipping {skip_docs:,} docs. Collecting new data...")
            continue

        streamed += 1
        text = example["text"]
        est_tokens = len(estimator.encode(text, add_special_tokens=False))

        if est_tokens < min_tokens:
            continue

        documents.append({
            "text": text,
            "uid": uid_offset + len(documents),
        })

        total_tokens += est_tokens
        pbar.update(est_tokens)

        if total_tokens >= target_tokens:
            break

        if len(documents) % 10000 == 0:
            pbar.set_postfix({"docs": len(documents)})

    pbar.close()
    return documents, total_tokens, streamed


def download_base_data(output_dir: Path) -> tuple[Path, int]:
    """Download the base 2B tokens and create train/val/test splits.

    This exactly reproduces the original notebook 01 procedure:
    - Stream ~2B tokens from FineWeb-Edu
    - Shuffle with seed=42
    - Split 95% train / 2.5% val / 2.5% test

    Returns:
        (output_dir, num_documents_streamed)
    """
    if (output_dir / "train").exists() and (output_dir / "test").exists():
        console.print(f"[yellow]Base raw data already exists at {output_dir}, skipping download.[/yellow]")
        train_ds = load_from_disk(str(output_dir / "train"))
        val_ds = load_from_disk(str(output_dir / "val"))
        test_ds = load_from_disk(str(output_dir / "test"))
        total_docs = len(train_ds) + len(val_ds) + len(test_ds)
        return output_dir, total_docs

    console.print("[green]Step 1: Downloading base 2B tokens from FineWeb-Edu...[/green]")
    estimator = AutoTokenizer.from_pretrained("gpt2")

    documents, total_tokens, total_streamed = stream_documents(
        estimator=estimator,
        target_tokens=BASE_TOKENS,
        min_tokens=BASE_MIN_TOKENS,
        desc="Downloading base data",
    )

    console.print(f"Collected {len(documents):,} documents ({total_tokens / 1e9:.2f}B tokens)")

    # Shuffle and split — exactly as in original notebook
    random.seed(BASE_SEED)
    random.shuffle(documents)

    n = len(documents)
    train_end = int(BASE_TRAIN_RATIO * n)
    val_end = int((BASE_TRAIN_RATIO + BASE_VAL_RATIO) * n)

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

    console.print(f"[green]Base raw data saved to {output_dir}[/green]")
    return output_dir, len(documents)


def download_extra_train_data(
    raw_data_dir: Path,
    extra_tokens: int,
    base_docs_streamed: int,
) -> None:
    """Download additional training documents beyond the base 2B.

    Resumes the FineWeb-Edu stream from where the base download left off,
    so there is no overlap with val/test data.
    """
    extra_train_path = raw_data_dir / "train_extra"
    if extra_train_path.exists():
        console.print(f"[yellow]Extra train data already exists at {extra_train_path}, skipping.[/yellow]")
        return

    console.print(f"[green]Step 2: Downloading ~{extra_tokens / 1e9:.0f}B additional training tokens...[/green]")
    console.print(f"  Skipping first {base_docs_streamed:,} documents (already used for base split)")

    estimator = AutoTokenizer.from_pretrained("gpt2")

    documents, total_tokens, _ = stream_documents(
        estimator=estimator,
        target_tokens=extra_tokens,
        min_tokens=BASE_MIN_TOKENS,
        skip_docs=base_docs_streamed,
        uid_offset=base_docs_streamed,
        desc="Downloading extra train data",
    )

    console.print(f"Collected {len(documents):,} extra documents ({total_tokens / 1e9:.2f}B tokens)")

    ds = Dataset.from_list(documents)
    ds.save_to_disk(str(extra_train_path))
    console.print(f"[green]Extra train data saved to {extra_train_path}[/green]")


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

        # For train: concatenate with extra train data if it exists
        if split_name == "train":
            extra_path = raw_data_dir / "train_extra"
            if extra_path.exists():
                console.print(f"[green]  Concatenating extra train data...[/green]")
                extra_ds = load_from_disk(str(extra_path))
                raw_ds = concatenate_datasets([raw_ds, extra_ds])
                console.print(f"    Combined train: {len(raw_ds):,} documents")

        tok_ds = tokenize_split(raw_ds, tokenizer, num_proc)

        total_tokens = sum(len(x) for x in tok_ds["input_ids"])
        console.print(f"    {len(tok_ds):,} docs, {total_tokens:,} tokens ({total_tokens / 1e9:.2f}B)")

        tok_ds.save_to_disk(str(output_dir / split_name))

    console.print(f"[green]Tokenized data saved to {output_dir}[/green]")


@app.command()
def main(
    tokenizer: list[str] = typer.Option(..., "--tokenizer", "-t", help="Path(s) to tokenizer(s)"),
    output_dir: Path = typer.Option("data", "--output-dir", "-o", help="Base output directory"),
    raw_data_dir: Optional[Path] = typer.Option(None, "--raw-data-dir", help="Existing raw data directory (skip download)"),
    target_tokens: int = typer.Option(2_000_000_000, "--target-tokens", help="Total target training tokens"),
    num_proc: int = typer.Option(8, "--num-proc", help="Number of processes for tokenization"),
) -> None:
    """Download FineWeb-Edu and tokenize for training.

    Val/test splits always come from the first 2B tokens (seed=42 shuffle),
    matching the original experiment setup. If --target-tokens > 2B, additional
    documents are streamed and added only to the train split.
    """
    # Step 1: Download base 2B tokens (creates val/test)
    if raw_data_dir and raw_data_dir.exists():
        console.print(f"[green]Using existing raw data from {raw_data_dir}[/green]")
        train_ds = load_from_disk(str(raw_data_dir / "train"))
        val_ds = load_from_disk(str(raw_data_dir / "val"))
        test_ds = load_from_disk(str(raw_data_dir / "test"))
        base_docs = len(train_ds) + len(val_ds) + len(test_ds)
    else:
        raw_data_dir = output_dir / "fineweb-edu-raw"
        raw_data_dir, base_docs = download_base_data(raw_data_dir)

    # Step 2: Download extra training data if needed
    # Base train is ~95% of 2B = ~1.9B tokens
    base_train_tokens = int(BASE_TOKENS * BASE_TRAIN_RATIO)
    if target_tokens > base_train_tokens:
        extra_tokens = target_tokens - base_train_tokens
        download_extra_train_data(raw_data_dir, extra_tokens, base_docs)

    # Step 3: Tokenize with each tokenizer
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
