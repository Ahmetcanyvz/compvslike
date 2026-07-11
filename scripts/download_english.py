"""Download raw English FineWeb-Edu, matching the ORIGINAL 2B-split convention.

This reproduces the original pipeline layout exactly:
  * val/test are carved from the FIRST ~2B tokens (seed-42 shuffle), giving
    47,384 documents each -> a single parquet shard each. These are the
    held-out sets the paper's BPB/BLiMP numbers are computed on.
  * train is the 95% split of that same first 2B (~1.8M docs), optionally
    extended with additional streamed documents (saved under train_extra/ as
    chunks) until --target-tokens total is reached. val/test never grow.

The extra-train phase is resumable (sharded + progress checkpoint). The base
2B phase is not resumable but takes only ~30 min; if interrupted, rerun.

Usage:
    python scripts/download_english.py -o data/fineweb-edu-raw --target-tokens 20000000000
"""

import json
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
BASE_TOKENS = 2_000_000_000  # first 2B define val/test, exactly as the original


def _extra_dir(output_dir: Path) -> Path:
    return output_dir / "train_extra"


def _progress_file(output_dir: Path) -> Path:
    return output_dir / "train_extra" / "_progress.json"


def _load_progress(output_dir: Path) -> dict:
    p = _progress_file(output_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {"stream_pos": 0, "tokens": 0, "docs": 0, "chunks": 0}


def _save_progress(output_dir: Path, prog: dict) -> None:
    _progress_file(output_dir).write_text(json.dumps(prog))


def build_base_2b(output_dir: Path, tokenizer, min_tokens: int) -> int:
    """Collect the first ~2B tokens, shuffle (seed 42), split 95/2.5/2.5, save.

    Returns the number of raw stream examples consumed to build the base
    (so the extra phase can skip exactly past them).
    """
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    documents = []
    total_tokens = 0
    stream_pos = 0
    pbar = tqdm(total=BASE_TOKENS, unit="tok", desc="Base 2B (defines val/test)")
    for example in dataset:
        stream_pos += 1
        text = example["text"]
        est = len(tokenizer.encode(text, add_special_tokens=False))
        if est < min_tokens:
            continue
        documents.append({"text": text, "uid": len(documents)})
        total_tokens += est
        pbar.update(est)
        if total_tokens >= BASE_TOKENS:
            break
    pbar.close()

    random.seed(SEED)
    random.shuffle(documents)
    n = len(documents)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    Dataset.from_list(documents[:train_end]).save_to_disk(str(output_dir / "train"))
    Dataset.from_list(documents[train_end:val_end]).save_to_disk(str(output_dir / "val"))
    Dataset.from_list(documents[val_end:]).save_to_disk(str(output_dir / "test"))
    print(
        f"Base: {n:,} docs -> train {train_end:,}  "
        f"val {val_end - train_end:,}  test {n - val_end:,}  (stream_pos={stream_pos:,})"
    )
    (output_dir / "_base_done.json").write_text(json.dumps({"stream_pos": stream_pos, "base_docs": n}))
    return stream_pos


@app.command()
def main(
    output_dir: Path = typer.Option("data/fineweb-edu-raw", "--output-dir", "-o"),
    target_tokens: int = typer.Option(20_000_000_000, "--target-tokens"),
    min_tokens: int = typer.Option(50, "--min-tokens"),
    estimator_tokenizer: str = typer.Option("gpt2", "--estimator"),
    batch_size: int = typer.Option(2000, "--batch-size", help="Docs per tokenizer batch (extra phase)"),
    chunk_docs: int = typer.Option(500_000, "--chunk-docs", help="Docs per train_extra chunk"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(estimator_tokenizer)

    # --- Phase A: base 2B defines train/val/test (val/test = 47,384 each) ---
    base_marker = output_dir / "_base_done.json"
    if base_marker.exists() and (output_dir / "test").exists():
        base_stream_pos = json.loads(base_marker.read_text())["stream_pos"]
        print(f"Base already built (stream_pos={base_stream_pos:,}); skipping phase A.")
    else:
        base_stream_pos = build_base_2b(output_dir, tokenizer, min_tokens)

    base_train_tokens = int(BASE_TOKENS * TRAIN_RATIO)
    if target_tokens <= base_train_tokens:
        print("Target <= base train tokens; no extra download needed. Done.")
        return

    # --- Phase B: extend train up to target via resumable train_extra chunks ---
    _extra_dir(output_dir).mkdir(parents=True, exist_ok=True)
    extra_target = target_tokens - base_train_tokens
    prog = _load_progress(output_dir)
    stream_pos = prog["stream_pos"] or base_stream_pos
    tokens = prog["tokens"]
    docs = prog["docs"]
    chunk_idx = prog["chunks"]
    print(f"Extra: resuming at {tokens/1e9:.2f}B/{extra_target/1e9:.2f}B, stream_pos={stream_pos:,}")

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    stream_it = iter(dataset)

    # Skip everything already consumed (base + any extra already saved).
    if stream_pos > 0:
        skip_pbar = tqdm(total=stream_pos, unit="doc", desc="Skipping consumed docs")
        seen = 0
        for _ in stream_it:
            seen += 1
            skip_pbar.update(1)
            if seen >= stream_pos:
                break
        skip_pbar.close()

    pbar = tqdm(total=extra_target, initial=tokens, unit="tok", desc="Extra train")
    text_buf: list[str] = []
    chunk_buf: list[dict] = []

    def flush_batch() -> None:
        nonlocal tokens, docs
        if not text_buf:
            return
        enc = tokenizer(text_buf, add_special_tokens=False)["input_ids"]
        for txt, ids in zip(text_buf, enc):
            if len(ids) < min_tokens:
                continue
            chunk_buf.append({"text": txt, "uid": base_stream_pos + docs})
            docs += 1
            tokens += len(ids)
            pbar.update(len(ids))
        text_buf.clear()

    def flush_chunk(force: bool = False) -> None:
        nonlocal chunk_idx
        if chunk_buf and (len(chunk_buf) >= chunk_docs or force):
            Dataset.from_list(chunk_buf).save_to_disk(str(_extra_dir(output_dir) / f"chunk_{chunk_idx:05d}"))
            chunk_idx += 1
            chunk_buf.clear()
            _save_progress(output_dir, {"stream_pos": stream_pos, "tokens": tokens, "docs": docs, "chunks": chunk_idx})

    for example in stream_it:
        stream_pos += 1
        text_buf.append(example["text"])
        if len(text_buf) >= batch_size:
            flush_batch()
            flush_chunk()
            if tokens >= extra_target:
                break

    flush_batch()
    flush_chunk(force=True)
    pbar.close()
    print(f"Extra done: {docs:,} docs, {tokens/1e9:.2f}B tokens in {chunk_idx} chunks")
    print(f"Total train ~= {(base_train_tokens + tokens)/1e9:.2f}B. val/test unchanged (47,384 each).")


if __name__ == "__main__":
    app()
