"""Download raw English FineWeb-Edu with train/val/test splits.

Streams FineWeb-Edu, collects documents until reaching target token count
(estimated with GPT-2 tokenizer), splits 95/2.5/2.5 into train/val/test,
and saves as HuggingFace Arrow datasets.

Usage:
    python scripts/download_english.py -o data/fineweb-edu-raw --target-tokens 20000000000
"""

import random
from pathlib import Path

import typer
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer()

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025


@app.command()
def main(
    output_dir: Path = typer.Option("data/fineweb-edu-raw", "--output-dir", "-o"),
    target_tokens: int = typer.Option(20_000_000_000, "--target-tokens"),
    min_tokens: int = typer.Option(50, "--min-tokens"),
    estimator_tokenizer: str = typer.Option("gpt2", "--estimator"),
) -> None:
    if output_dir.exists() and (output_dir / "train").exists():
        print(f"{output_dir} already has a train split, skipping.")
        raise typer.Exit()

    tokenizer = AutoTokenizer.from_pretrained(estimator_tokenizer)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    documents = []
    total_tokens = 0
    pbar = tqdm(total=target_tokens, unit="tok", desc="Collecting English")

    for example in dataset:
        text = example["text"]
        est_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if est_tokens < min_tokens:
            continue

        documents.append({"text": text, "uid": len(documents)})
        total_tokens += est_tokens
        pbar.update(est_tokens)
        if total_tokens >= target_tokens:
            break
        if len(documents) % 10000 == 0:
            pbar.set_postfix({"docs": len(documents)})

    pbar.close()
    print(f"Collected {len(documents):,} documents, {total_tokens/1e9:.2f}B tokens")

    random.seed(SEED)
    random.shuffle(documents)

    n = len(documents)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    train_docs = documents[:train_end]
    val_docs = documents[train_end:val_end]
    test_docs = documents[val_end:]

    print(f"Train: {len(train_docs):,}  Val: {len(val_docs):,}  Test: {len(test_docs):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_docs).save_to_disk(str(output_dir / "train"))
    Dataset.from_list(val_docs).save_to_disk(str(output_dir / "val"))
    Dataset.from_list(test_docs).save_to_disk(str(output_dir / "test"))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    app()
