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
        writer_batch_size=5000,
    )


@app.command()
def main(
    tokenizer: list[str] = typer.Option(..., "--tokenizer", "-t", help="Path(s) to tokenizer(s)"),
    output_dir: Path = typer.Option("data/multilingual", "--output-dir", "-o"),
    raw_data_dir: Optional[Path] = typer.Option(None, "--raw-data-dir", help="Existing multilingual raw data dir (skip download)"),
    total_tokens: int = typer.Option(2_000_000_000, "--total-tokens"),
    num_proc: int = typer.Option(8, "--num-proc"),
    eng_raw_dir: Optional[Path] = typer.Option(None, "--eng-raw-dir", help="Existing English raw data dir (reuse val/test)"),
    merge_only: bool = typer.Option(False, "--merge-only", help="Do download-check + merge, then stop (no tokenization)"),
    tokenize_only: bool = typer.Option(False, "--tokenize-only", help="Skip merge creation; tokenize the given tokenizer(s). Merge must already exist."),
) -> None:
    """Download multilingual data and tokenize.

    For parallel tokenization: run once with --merge-only, then run a job per
    tokenizer with --tokenize-only (all reading the shared merged data).
    """

    raw_dir = raw_data_dir if raw_data_dir and raw_data_dir.exists() else output_dir / "raw"
    merge_dir = output_dir / "merged"

    # Build per-language raw data paths
    lang_raw_paths = {}
    for lang in LANGUAGES:
        if lang == "eng" and eng_raw_dir and eng_raw_dir.exists():
            lang_raw_paths[lang] = eng_raw_dir
            console.print(f"  {lang}: using existing data from {eng_raw_dir}")
        elif (raw_dir / lang).exists():
            lang_raw_paths[lang] = raw_dir / lang
            console.print(f"  {lang}: using existing data from {raw_dir / lang}")
        else:
            # Download if not found
            lang_tokens = int(total_tokens * LANGUAGES[lang][2])
            download_language(lang, LANGUAGES[lang][0], LANGUAGES[lang][1], lang_tokens, raw_dir)
            lang_raw_paths[lang] = raw_dir / lang

    # Step 2: Merge train and val (test stays per-language for separate evaluation)
    console.print("\n[bold]Step 2: Merging languages (train + val only, test stays per-language)[/bold]")
    from datasets import concatenate_datasets

    for split in ["train", "val"]:
        merged_split_dir = merge_dir / split
        if merged_split_dir.exists():
            console.print(f"[yellow]Merged {split} already exists, skipping[/yellow]")
            continue

        all_docs = []
        for lang, lang_path in lang_raw_paths.items():
            lang_split = lang_path / split
            if lang_split.exists():
                ds = load_from_disk(str(lang_split))

                # For English train, also load train_extra chunks to get ~10B tokens
                if lang == "eng" and split == "train":
                    extra_dir = lang_path / "train_extra"
                    if extra_dir.exists():
                        chunk_paths = sorted(extra_dir.glob("chunk_*"))
                        if chunk_paths:
                            console.print(f"  {lang}/{split}: {len(ds):,} docs + loading extra chunks...")
                            extra_datasets = [load_from_disk(str(p)) for p in chunk_paths]
                            ds = concatenate_datasets([ds] + extra_datasets)
                            # Take ~50% to get ~10B tokens from ~20B total
                            target_docs = len(ds) // 2
                            ds = ds.select(range(target_docs))
                            console.print(f"  {lang}/{split}: trimmed to {len(ds):,} docs (~10B tokens)")
                else:
                    console.print(f"  {lang}/{split}: {len(ds):,} documents")
                all_docs.append(ds)

        if all_docs:
            merged = concatenate_datasets(all_docs)
            merged = merged.shuffle(seed=SEED)
            merged_split_dir.parent.mkdir(parents=True, exist_ok=True)
            merged.save_to_disk(str(merged_split_dir))
            console.print(f"  Merged {split}: {len(merged):,} documents")

    if merge_only:
        console.print("\n[green bold]Merge complete (--merge-only); skipping tokenization.[/green bold]")
        return

    if tokenize_only and not (merge_dir / "train").exists():
        raise typer.BadParameter(
            "--tokenize-only requires the merged data to exist; run with --merge-only first."
        )

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

        # Tokenize merged train and val
        for split in ["train", "val"]:
            merged_path = merge_dir / split
            if not merged_path.exists():
                console.print(f"  [yellow]{split} not found, skipping[/yellow]")
                continue

            console.print(f"  Tokenizing {split}...")
            raw_ds = load_from_disk(str(merged_path))
            tok_ds = tokenize_split(raw_ds, tok, num_proc)

            tok_ds.save_to_disk(str(tok_output / split))
            total_toks = sum(sum(len(x) for x in batch["input_ids"]) for batch in tok_ds.iter(batch_size=1000))
            console.print(f"    {len(tok_ds):,} docs, {total_toks:,} tokens ({total_toks / 1e9:.2f}B)")

        # Tokenize test per-language (for separate evaluation)
        for lang, lang_path in lang_raw_paths.items():
            lang_test = lang_path / "test"
            if not lang_test.exists():
                continue

            test_out = tok_output / f"test_{lang}"
            if test_out.exists():
                console.print(f"  [yellow]test_{lang} already tokenized, skipping[/yellow]")
                continue

            console.print(f"  Tokenizing test_{lang}...")
            raw_ds = load_from_disk(str(lang_test))
            tok_ds = tokenize_split(raw_ds, tok, num_proc)

            total_toks = sum(sum(len(x) for x in batch["input_ids"]) for batch in tok_ds.iter(batch_size=1000))
            console.print(f"    {len(tok_ds):,} docs, {total_toks:,} tokens ({total_toks / 1e9:.2f}B)")

            tok_ds.save_to_disk(str(test_out))

    console.print(f"\n[green bold]All done![/green bold]")


if __name__ == "__main__":
    app()
