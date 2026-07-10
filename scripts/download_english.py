"""Download raw English FineWeb-Edu with resumable, sharded streaming.

Streams FineWeb-Edu, estimates token counts with a batched GPT-2 tokenizer,
and writes documents to disk in shards as it goes. Progress is checkpointed
after every shard, so a re-submit resumes from the last saved shard instead
of restarting. Once the token target is reached, all shards are concatenated
(memory-mapped), shuffled, split 95/2.5/2.5, and saved as train/val/test.

Resume: just re-run with the same --output-dir. It reads _progress.json,
skips the already-consumed stream positions, and continues.

Usage:
    python scripts/download_english.py -o data/fineweb-edu-raw --target-tokens 20000000000
"""

import json
import shutil
from pathlib import Path

import typer
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer

app = typer.Typer()

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025


def _shard_dir(output_dir: Path) -> Path:
    return output_dir / "_shards"


def _progress_file(output_dir: Path) -> Path:
    return output_dir / "_progress.json"


def _load_progress(output_dir: Path) -> dict:
    p = _progress_file(output_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"stream_pos": 0, "tokens": 0, "docs": 0, "shards": 0}


def _save_progress(output_dir: Path, prog: dict) -> None:
    _progress_file(output_dir).write_text(json.dumps(prog))


@app.command()
def main(
    output_dir: Path = typer.Option("data/fineweb-edu-raw", "--output-dir", "-o"),
    target_tokens: int = typer.Option(20_000_000_000, "--target-tokens"),
    min_tokens: int = typer.Option(50, "--min-tokens"),
    estimator_tokenizer: str = typer.Option("gpt2", "--estimator"),
    batch_size: int = typer.Option(2000, "--batch-size", help="Docs per tokenizer batch"),
    shard_docs: int = typer.Option(500_000, "--shard-docs", help="Docs per saved shard"),
    keep_shards: bool = typer.Option(False, "--keep-shards", help="Keep shards after assembly"),
) -> None:
    if (output_dir / "train").exists():
        print(f"{output_dir}/train already exists, nothing to do.")
        raise typer.Exit()

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = _shard_dir(output_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(estimator_tokenizer)

    prog = _load_progress(output_dir)
    stream_pos = prog["stream_pos"]
    tokens = prog["tokens"]
    docs = prog["docs"]
    shard_idx = prog["shards"]
    print(
        f"Resuming: {docs:,} docs, {tokens/1e9:.2f}B tokens, "
        f"stream_pos={stream_pos:,}, shards={shard_idx}"
    )

    # Collection phase (skipped entirely if we already hit the target).
    if tokens < target_tokens:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        stream_it = iter(dataset)

        # Skip already-consumed stream positions (fast — no tokenization).
        if stream_pos > 0:
            skip_pbar = tqdm(total=stream_pos, unit="doc", desc="Skipping seen docs")
            seen = 0
            for _ in stream_it:
                seen += 1
                skip_pbar.update(1)
                if seen >= stream_pos:
                    break
            skip_pbar.close()

        pbar = tqdm(total=target_tokens, initial=tokens, unit="tok", desc="Collecting English")
        text_buf: list[str] = []
        shard_buf: list[dict] = []

        def flush_batch() -> None:
            nonlocal tokens, docs
            if not text_buf:
                return
            encoded = tokenizer(text_buf, add_special_tokens=False)["input_ids"]
            for txt, ids in zip(text_buf, encoded):
                n = len(ids)
                if n < min_tokens:
                    continue
                shard_buf.append({"text": txt, "uid": docs})
                docs += 1
                tokens += n
                pbar.update(n)
            text_buf.clear()

        def flush_shard(force: bool = False) -> None:
            nonlocal shard_idx
            if shard_buf and (len(shard_buf) >= shard_docs or force):
                Dataset.from_list(shard_buf).save_to_disk(str(shard_dir / f"shard_{shard_idx:05d}"))
                shard_idx += 1
                shard_buf.clear()
                _save_progress(
                    output_dir,
                    {"stream_pos": stream_pos, "tokens": tokens, "docs": docs, "shards": shard_idx},
                )

        for example in stream_it:
            stream_pos += 1
            text_buf.append(example["text"])
            if len(text_buf) >= batch_size:
                flush_batch()
                flush_shard()
                if tokens >= target_tokens:
                    break

        flush_batch()
        flush_shard(force=True)
        pbar.close()
        print(f"Collected {docs:,} docs, {tokens/1e9:.2f}B tokens across {shard_idx} shards")

    # Assembly phase: concat shards (memory-mapped), shuffle, split, save.
    shard_paths = sorted(shard_dir.glob("shard_*"))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {shard_dir}")

    full = concatenate_datasets([load_from_disk(str(p)) for p in shard_paths])
    full = full.shuffle(seed=SEED)

    n = len(full)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    full.select(range(0, train_end)).save_to_disk(str(output_dir / "train"))
    full.select(range(train_end, val_end)).save_to_disk(str(output_dir / "val"))
    full.select(range(val_end, n)).save_to_disk(str(output_dir / "test"))
    print(f"Train {train_end:,}  Val {val_end - train_end:,}  Test {n - val_end:,}")

    if not keep_shards:
        shutil.rmtree(shard_dir)
        _progress_file(output_dir).unlink(missing_ok=True)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    app()
