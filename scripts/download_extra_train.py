"""Download additional training data from FineWeb-Edu.

Resumes streaming from where the original 01_prepare_fineweb.ipynb left off
(~1.9M documents) and collects new documents until reaching the target token count.
Only extends the training set — val/test splits are NOT touched.

Usage:
    uv run python scripts/download_extra_train.py \
        --output-dir data/fineweb-edu-raw-20B \
        --target-tokens 20_000_000_000 \
        --skip-docs 1_895_346
"""

from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import typer

app = typer.Typer()


@app.command()
def main(
    output_dir: Path = typer.Option("data/fineweb-edu-raw-20B", help="Output directory for extended train split"),
    target_tokens: int = typer.Option(20_000_000_000, help="Target total tokens (including existing ~2B)"),
    existing_train: Path = typer.Option(None, help="Path to existing train split (to concatenate with)"),
    skip_docs: int = typer.Option(1_895_346, help="Number of docs to skip (already downloaded in original run)"),
    min_tokens: int = typer.Option(50, help="Minimum tokens per document"),
    save_every: int = typer.Option(1_000_000, help="Save checkpoint every N documents"),
    estimator_tokenizer: str = typer.Option("gpt2", help="Tokenizer for estimating token counts"),
) -> None:
    """Download additional FineWeb-Edu documents for training."""

    # Existing data has ~2B tokens
    existing_tokens = 2_000_000_000
    extra_tokens_needed = target_tokens - existing_tokens

    if extra_tokens_needed <= 0:
        print(f"Already have {existing_tokens:,} tokens, target is {target_tokens:,}. Nothing to do.")
        raise typer.Exit()

    print(f"Existing tokens: ~{existing_tokens / 1e9:.1f}B")
    print(f"Target tokens:    {target_tokens / 1e9:.1f}B")
    print(f"Need to collect: ~{extra_tokens_needed / 1e9:.1f}B additional tokens")
    print(f"Skipping first {skip_docs:,} documents from stream")
    print()

    # Load estimator tokenizer
    tokenizer = AutoTokenizer.from_pretrained(estimator_tokenizer)

    # Stream FineWeb-Edu
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    # Collect new documents
    new_documents = []
    new_tokens = 0
    skipped = 0

    pbar_skip = tqdm(total=skip_docs, unit="doc", desc="Skipping existing docs")
    pbar_collect = tqdm(total=extra_tokens_needed, unit="tok", desc="Collecting new data")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # UID offset: continue from where existing data left off
    uid_offset = skip_docs

    for i, example in enumerate(dataset):
        # Skip documents that were already downloaded
        if skipped < skip_docs:
            skipped += 1
            pbar_skip.update(1)
            if skipped == skip_docs:
                pbar_skip.close()
                print(f"\nDone skipping. Starting collection from document {i}...")
            continue

        text = example["text"]
        est_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        # Skip very short documents
        if est_tokens < min_tokens:
            continue

        new_documents.append({
            "text": text,
            "uid": uid_offset + len(new_documents),
        })

        new_tokens += est_tokens
        pbar_collect.update(est_tokens)

        if new_tokens >= extra_tokens_needed:
            break

        if len(new_documents) % 10000 == 0:
            pbar_collect.set_postfix({"docs": len(new_documents)})

        # Periodic checkpoint
        if save_every > 0 and len(new_documents) % save_every == 0:
            checkpoint_path = output_dir / "train_new_checkpoint"
            print(f"\n  Checkpoint: {len(new_documents):,} docs, {new_tokens / 1e9:.2f}B tokens")
            ds = Dataset.from_list(new_documents)
            ds.save_to_disk(str(checkpoint_path))

    pbar_collect.close()

    print(f"\nCollected {len(new_documents):,} new documents")
    print(f"New tokens: {new_tokens:,} ({new_tokens / 1e9:.2f}B)")
    print(f"Total tokens (existing + new): ~{(existing_tokens + new_tokens) / 1e9:.2f}B")

    # Save new documents
    print("\nSaving new documents...")
    new_ds = Dataset.from_list(new_documents)

    if existing_train and Path(existing_train).exists():
        # Concatenate with existing train split
        print(f"Loading existing train split from {existing_train}...")
        from datasets import concatenate_datasets

        old_ds = load_from_disk(str(existing_train))
        combined_ds = concatenate_datasets([old_ds, new_ds])
        save_path = output_dir / "train"
        combined_ds.save_to_disk(str(save_path))
        print(f"Saved combined train split ({len(combined_ds):,} docs) to {save_path}")
    else:
        # Save new documents only
        save_path = output_dir / "train_new"
        new_ds.save_to_disk(str(save_path))
        print(f"Saved new documents ({len(new_ds):,} docs) to {save_path}")
        print("\nTo combine with existing train data, re-run with:")
        print(f"  --existing-train <path_to_existing_train>")

    # Clean up checkpoint if it exists
    checkpoint_path = output_dir / "train_new_checkpoint"
    if checkpoint_path.exists():
        import shutil
        shutil.rmtree(checkpoint_path)
        print("Cleaned up checkpoint.")


if __name__ == "__main__":
    app()
