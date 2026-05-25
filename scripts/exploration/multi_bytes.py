"""Sum UTF-8 byte counts over the multilingual held-out test corpus.

The output is corpus-only (independent of tokenizer); divide by each
tokenizer's total token count to get its BPT on the multi test set.
"""
import os
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(os.environ.get("LM_TRAINER_ROOT", Path(__file__).resolve().parent.parent.parent))
RAW = ROOT / "data" / "multilingual-raw"
LANGS = ["eng", "deu", "spa", "tur", "cmn"]

if not RAW.exists():
    raise SystemExit(f"Multilingual raw dir not found at {RAW}")

grand_total = 0
print(f"{'Split':<10} {'docs':>10} {'bytes':>16}")
print("-" * 40)
for lang in LANGS:
    split_dir = RAW / f"test_{lang}"
    if not split_dir.exists():
        print(f"[skip] {split_dir} not found")
        continue
    ds = load_from_disk(str(split_dir))
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    n_bytes = 0
    for doc in ds[col]:
        n_bytes += len(doc.encode("utf-8"))
    print(f"test_{lang:<5} {len(ds):>10,} {n_bytes:>16,}")
    grand_total += n_bytes

print("-" * 40)
print(f"{'TOTAL':<10} {'':>10} {grand_total:>16,}")
print()
print("BPT = N_bytes / N_tokens.  Plug into:")
for name, tokens in [
    ("BPE", 172_372_100),
    ("CompMax", 217_983_502),
    ("GreedyLL", 173_917_215),
    ("UnigramLM", 240_669_745),
]:
    print(f"  {name:<10} BPT = {grand_total / tokens:.4f}")
