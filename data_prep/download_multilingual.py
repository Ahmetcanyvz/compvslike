"""Download multilingual data from FineWeb-2.

Downloads German, Spanish, Turkish, Chinese from FineWeb-2.
English data is reused from existing fineweb-edu-raw.

Usage:
    python scripts/download_multilingual.py -o data/multilingual-raw
"""

import random
from pathlib import Path

import typer
from datasets import Dataset, load_dataset
from rich.console import Console
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer()
console = Console()

LANGUAGES = {
    "deu": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
    "spa": ("HuggingFaceFW/fineweb-2", "spa_Latn"),
    "tur": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
    "cmn": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
}

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025


@app.command()
def main(
    output_dir: Path = typer.Option("data/multilingual-raw", "--output-dir", "-o"),
    tokens_per_lang: int = typer.Option(2_500_000_000, "--tokens-per-lang", help="Tokens per language"),
    min_tokens: int = typer.Option(50, "--min-tokens"),
    lang: str = typer.Option(None, "--lang", help="Download only this language (deu/spa/tur/cmn). Default: all."),
    revision: str = typer.Option("main", "--revision", help="Pin the HF dataset revision for reproducibility"),
) -> None:
    """Download German, Spanish, Turkish, Chinese from FineWeb-2."""

    if lang is not None and lang not in LANGUAGES:
        raise typer.BadParameter(f"Unknown language '{lang}'. Choose from: {', '.join(LANGUAGES)}")

    selected = {lang: LANGUAGES[lang]} if lang is not None else LANGUAGES

    estimator = AutoTokenizer.from_pretrained("gpt2")

    for lang, (dataset_name, config_name) in selected.items():
        train_dir = output_dir / lang / "train"
        test_dir = output_dir / lang / "test"

        if train_dir.exists() and test_dir.exists():
            console.print(f"[yellow]{lang}: already downloaded, skipping[/yellow]")
            continue

        console.print(f"[green]{lang}: Downloading ~{tokens_per_lang / 1e9:.2f}B tokens from {dataset_name} ({config_name})[/green]")

        dataset = load_dataset(dataset_name, config_name, split="train", streaming=True, revision=revision)

        documents = []
        total_tokens = 0

        pbar = tqdm(total=tokens_per_lang, unit="tok", desc=f"{lang}")

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

            if total_tokens >= tokens_per_lang:
                break

            if len(documents) % 10000 == 0:
                pbar.set_postfix({"docs": len(documents)})

        pbar.close()
        console.print(f"{lang}: {len(documents):,} documents ({total_tokens / 1e9:.2f}B tokens)")

        # Shuffle and split
        random.seed(SEED)
        random.shuffle(documents)

        n = len(documents)
        train_end = int(TRAIN_RATIO * n)
        val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

        splits = {
            "train": documents[:train_end],
            "val": documents[train_end:val_end],
            "test": documents[val_end:],
        }

        for split_name, docs in splits.items():
            split_dir = output_dir / lang / split_name
            split_dir.parent.mkdir(parents=True, exist_ok=True)
            ds = Dataset.from_list(docs)
            ds.save_to_disk(str(split_dir))
            console.print(f"  {split_name}: {len(docs):,} documents")

    console.print(f"\n[green bold]All downloads complete![/green bold]")
    console.print(f"Raw data saved to {output_dir}/")
    console.print(f"English data should be reused from existing fineweb-edu-raw/")


if __name__ == "__main__":
    app()
