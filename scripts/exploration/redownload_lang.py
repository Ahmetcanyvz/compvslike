"""Re-download a single language and re-create its train/val/test splits.

Useful when one language's `<raw>/<lang>/` directory was deleted but you don't
want to re-run the full multilingual prep. Output matches the layout
prepare_multilingual.py writes (same SEED / ratios) so the regenerated test
split is byte-identical to the original.
"""
import argparse
import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

SEED = 42
TRAIN_RATIO = 0.95
VAL_RATIO = 0.025

LANGUAGES = {
    "eng": ("HuggingFaceFW/fineweb-edu", None),
    "deu": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
    "spa": ("HuggingFaceFW/fineweb-2", "spa_Latn"),
    "tur": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
    "cmn": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(LANGUAGES))
    ap.add_argument("--target-tokens", type=int, default=2_500_000_000,
                    help="Match the size the original prep targeted for this language (default 2.5B = 12.5%% of a 20B-token multilingual corpus)")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to $LM_TRAINER_ROOT/data/multilingual-raw")
    ap.add_argument("--min-tokens", type=int, default=50)
    args = ap.parse_args()

    root = Path(os.environ.get("LM_TRAINER_ROOT", Path(__file__).resolve().parent.parent.parent))
    out_root = Path(args.output_dir) if args.output_dir else root / "data" / "multilingual-raw"

    name, config = LANGUAGES[args.lang]
    target_tokens = args.target_tokens

    print(f"Re-downloading {args.lang} from {name} (config={config}) — target {target_tokens / 1e9:.2f}B tokens")
    estimator = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset(name, config, split="train", streaming=True) if config \
        else load_dataset(name, split="train", streaming=True)

    documents = []
    total_tokens = 0
    pbar = tqdm(total=target_tokens, unit="tok", desc=args.lang)
    for example in ds:
        text = example["text"]
        est = len(estimator.encode(text, add_special_tokens=False))
        if est < args.min_tokens:
            continue
        documents.append({"text": text, "uid": len(documents)})
        total_tokens += est
        pbar.update(est)
        if total_tokens >= target_tokens:
            break
    pbar.close()
    print(f"Collected {len(documents):,} documents ({total_tokens / 1e9:.2f}B tokens)")

    random.seed(SEED)
    random.shuffle(documents)
    n = len(documents)
    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)
    splits = {
        "train": documents[:train_end],
        "val":   documents[train_end:val_end],
        "test":  documents[val_end:],
    }
    for split, docs in splits.items():
        out = out_root / args.lang / split
        out.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_list(docs).save_to_disk(str(out))
        print(f"  {split}: {len(docs):,} -> {out}")


if __name__ == "__main__":
    main()
