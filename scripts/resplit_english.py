"""Re-split an existing fineweb-edu-raw into the ORIGINAL 2B-based splits.

No downloading. Fixes a raw dir that was mistakenly split over the full ~20B
(giving oversized val/test) back to the original convention:

  * val/test = the 2.5%/2.5% slices of the FIRST ~2B docs (uid < BASE_DOCS),
    seed-42 shuffled -> 47,384 docs each (one parquet), matching the paper.
  * train    = the 95% slice of that same first 2B, plus every remaining doc
    (uid >= BASE_DOCS) saved under train_extra/ as chunks.

Relies on the `uid` field, which encodes original stream/collection order.

Usage:
    python scripts/resplit_english.py --raw-dir data/fineweb-edu-raw
"""

import random
from pathlib import Path

import typer
from datasets import Dataset, concatenate_datasets, load_from_disk

app = typer.Typer()

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025
BASE_DOCS = 1_895_346  # exact count from the original 2B download


@app.command()
def main(
    raw_dir: Path = typer.Option(..., "--raw-dir", help="Existing fineweb-edu-raw to re-split (in place)"),
    base_docs: int = typer.Option(BASE_DOCS, "--base-docs", help="Doc count that defined the original 2B"),
    chunk_docs: int = typer.Option(500_000, "--chunk-docs", help="Docs per train_extra chunk"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only report counts; write nothing"),
) -> None:
    parts = []
    for s in ["train", "val", "test"]:
        p = raw_dir / s
        if p.exists():
            parts.append(load_from_disk(str(p)))
    # include any existing train_extra chunks too, so nothing is lost
    extra_dir = raw_dir / "train_extra"
    if extra_dir.exists():
        for c in sorted(extra_dir.glob("chunk_*")):
            parts.append(load_from_disk(str(c)))
    if not parts:
        raise typer.BadParameter(f"No splits found under {raw_dir}")

    full = concatenate_datasets(parts)
    if "uid" not in full.column_names:
        raise typer.BadParameter("Data has no 'uid' column; cannot reconstruct original order.")

    total = len(full)
    print(f"Loaded {total:,} docs total from {raw_dir}")

    base = full.filter(lambda x: x["uid"] < base_docs, num_proc=8).sort("uid")
    extra = full.filter(lambda x: x["uid"] >= base_docs, num_proc=8)
    print(f"Base (uid < {base_docs:,}): {len(base):,} docs   |   extra: {len(extra):,} docs")

    if len(base) != base_docs:
        print(
            f"[WARN] base count {len(base):,} != expected {base_docs:,}. "
            "val/test may not byte-match the original. Check the raw source."
        )

    # Reproduce the original notebook: seed-42 Python shuffle of the base list.
    docs = base.to_list()
    random.seed(SEED)
    random.shuffle(docs)

    n = len(docs)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)
    print(
        f"Re-split -> train {train_end:,}  val {val_end - train_end:,}  test {n - val_end:,}"
        f"   (+ {len(extra):,} extra docs into train_extra/)"
    )

    if dry_run:
        print("[dry-run] nothing written.")
        return

    # Write to temp dirs first, then swap, so a failure can't corrupt the raw.
    tmp = raw_dir / "_resplit_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(docs[:train_end]).save_to_disk(str(tmp / "train"))
    Dataset.from_list(docs[train_end:val_end]).save_to_disk(str(tmp / "val"))
    Dataset.from_list(docs[val_end:]).save_to_disk(str(tmp / "test"))

    # extra -> train_extra chunks
    tmp_extra = tmp / "train_extra"
    tmp_extra.mkdir(parents=True, exist_ok=True)
    if len(extra) > 0:
        for i, start in enumerate(range(0, len(extra), chunk_docs)):
            extra.select(range(start, min(start + chunk_docs, len(extra)))).save_to_disk(
                str(tmp_extra / f"chunk_{i:05d}")
            )

    # Swap old splits out, move new ones in.
    import shutil

    for s in ["train", "val", "test", "train_extra"]:
        old = raw_dir / s
        if old.exists():
            shutil.rmtree(old)
    for s in ["train", "val", "test", "train_extra"]:
        src = tmp / s
        if src.exists():
            src.rename(raw_dir / s)
    shutil.rmtree(tmp)
    print(f"Done. val/test are now {val_end - train_end:,}/{n - val_end:,} docs (single parquet each).")


if __name__ == "__main__":
    app()
