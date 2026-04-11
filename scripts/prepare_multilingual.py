"""Download and prepare multilingual training data.

Downloads from FineWeb-Edu (English) and FineWeb-2 (other languages).
Creates train/val/test splits per language, then tokenizes with specified tokenizers.

Usage:
    python scripts/prepare_multilingual.py \
        -t tokenizers/bpe-128k \
        -o data/multilingual \
        --total-tokens 2000000000
"""

import random
from pathlib import Path
from typing import Optional

import typer
from datasets import Dataset, load_dataset, load_from_disk
from rich.console import Console
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer()
console = Console()

# Language configs: (dataset_name, config_name, percentage)
LANGUAGES = {
    "eng": ("HuggingFaceFW/fineweb-edu", None, 0.50),
    "deu": ("HuggingFaceFW/fineweb-2", "deu_Latn", 0.125),
    "spa": ("HuggingFaceFW/fineweb-2", "spa_Latn", 0.125),
    "tur": ("HuggingFaceFW/fineweb-2", "tur_Latn", 0.125),
    "cmn": ("HuggingFaceFW/fineweb-2", "cmn_Hani", 0.125),
}

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
MIN_TOKENS = 50


def download_language(
    lang: str,
    dataset_name: str,
    config_name: Optional[str],
    target_tokens: int,
    output_dir: Path,
    min_tokens: int = 50,
) -> None:
    """Download raw text for a single language."""
    train_dir = output_dir / lang / "train"
    test_dir = output_dir / lang / "test"

    if train_dir.exists() and test_dir.exists():
        console.print(f"[yellow]{lang}: already downloaded, skipping[/yellow]")
        return

    console.print(f"[green]{lang}: Downloading ~{target_tokens / 1e9:.2f}B tokens from {dataset_name}[/green]")

    estimator = AutoTokenizer.from_pretrained("gpt2")

    if config_name:
        dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)

    documents = []
    total_tokens = 0

    pbar = tqdm(total=target_tokens, unit="tok", desc=f"{lang}")

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
    console.print(f"{lang}: Collected {len(documents):,} documents ({total_tokens / 1e9:.2f}B tokens)")

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


def tokenize_split(raw_dataset, tokenizer, num_proc):
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


@app.command()
def main(
    tokenizer: list[str] = typer.Option(..., "--tokenizer", "-t", help="Path(s) to tokenizer(s)"),
    output_dir: Path = typer.Option("data/multilingual", "--output-dir", "-o"),
    total_tokens: int = typer.Option(2_000_000_000, "--total-tokens"),
    num_proc: int = typer.Option(8, "--num-proc"),
    eng_raw_dir: Optional[Path] = typer.Option(None, "--eng-raw-dir", help="Existing English raw data dir (reuse val/test)"),
) -> None:
    """Download multilingual data and tokenize."""

    raw_dir = output_dir / "raw"

    # Step 1: Download raw data per language
    console.print("[bold]Step 1: Downloading raw data[/bold]")
    for lang, (dataset_name, config_name, pct) in LANGUAGES.items():
        lang_tokens = int(total_tokens * pct)

        if lang == "eng" and eng_raw_dir and eng_raw_dir.exists():
            # Reuse existing English val/test, download only train
            eng_out = raw_dir / "eng"
            if not eng_out.exists():
                eng_out.mkdir(parents=True, exist_ok=True)
                # Copy val/test from existing
                import shutil
                for split in ["val", "test"]:
                    src = eng_raw_dir / split
                    dst = eng_out / split
                    if src.exists() and not dst.exists():
                        shutil.copytree(str(src), str(dst))
                        console.print(f"  Copied English {split} from {src}")

            # Download English train with target tokens
            train_dir = eng_out / "train"
            if not train_dir.exists():
                download_language("eng", dataset_name, config_name, lang_tokens, raw_dir)
        else:
            download_language(lang, dataset_name, config_name, lang_tokens, raw_dir)

    # Step 2: Merge all languages into combined train/val/test
    console.print("\n[bold]Step 2: Merging languages[/bold]")
    from datasets import concatenate_datasets

    for split in ["train", "val", "test"]:
        merged_dir = raw_dir / "merged" / split
        if merged_dir.exists():
            console.print(f"[yellow]Merged {split} already exists, skipping[/yellow]")
            continue

        all_docs = []
        for lang in LANGUAGES:
            lang_split = raw_dir / lang / split
            if lang_split.exists():
                ds = load_from_disk(str(lang_split))
                console.print(f"  {lang}/{split}: {len(ds):,} documents")
                all_docs.append(ds)

        if all_docs:
            merged = concatenate_datasets(all_docs)
            # Shuffle the merged dataset
            merged = merged.shuffle(seed=SEED)
            merged_dir.parent.mkdir(parents=True, exist_ok=True)
            merged.save_to_disk(str(merged_dir))
            console.print(f"  Merged {split}: {len(merged):,} documents")

    # Step 3: Tokenize with each tokenizer
    console.print("\n[bold]Step 3: Tokenizing[/bold]")
    for tok_path in tokenizer:
        tok_name = Path(tok_path).name
        tok_output = output_dir / f"multilingual-{tok_name}"

        if (tok_output / "train").exists():
            console.print(f"[yellow]{tok_name}: already tokenized, skipping[/yellow]")
            continue

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Tokenizer: {tok_name}[/bold]")
        console.print(f"{'=' * 60}")

        tok = AutoTokenizer.from_pretrained(tok_path)
        console.print(f"  Vocab size: {len(tok):,}")

        tok_output.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            merged_path = raw_dir / "merged" / split
            if not merged_path.exists():
                console.print(f"  [yellow]{split} not found, skipping[/yellow]")
                continue

            console.print(f"  Tokenizing {split}...")
            raw_ds = load_from_disk(str(merged_path))
            tok_ds = tokenize_split(raw_ds, tok, num_proc)

            total_toks = sum(len(x) for x in tok_ds["input_ids"])
            console.print(f"    {len(tok_ds):,} docs, {total_toks:,} tokens ({total_toks / 1e9:.2f}B)")

            tok_ds.save_to_disk(str(tok_output / split))

    console.print(f"\n[green bold]All done![/green bold]")


if __name__ == "__main__":
    app()
