"""Extension of kill_ratio: counts killed tokens with zero/low usage on test data."""
import argparse
import json
from collections import Counter

from datasets import load_from_disk

from config import TOKENIZER_PATHS, DATA_PATHS, SHORT, BPE_METHODS


def split_merge(mg):
    if isinstance(mg, str):
        return mg.split(" ", 1)
    return mg[0], mg[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizers", nargs="+", default=BPE_METHODS, choices=BPE_METHODS)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print("=" * 70)
    print("KILLED-TOKEN USAGE STATISTICS")
    print("=" * 70)
    print()
    print(f"  {'Method':<20} {'killed-zero':>12} {'killed-low(≤10)':>16} {'killed-zero %vocab':>20} {'survived-zero':>15}")
    print("  " + "-" * 90)

    for name in args.tokenizers:
        tok_path = TOKENIZER_PATHS[name]
        with open(f"{tok_path}/tokenizer.json") as f:
            model = json.load(f)["model"]
        vocab = model["vocab"]  # token -> id
        merges = model["merges"]

        # Created: tokens that result from a merge
        created = set()
        for mg in merges:
            a, b = split_merge(mg)
            created.add(a + b)

        # Consumed: tokens that get used as either side of a later merge,
        # AND were themselves created by an earlier merge.
        consumed = set()
        for mg in merges:
            a, b = split_merge(mg)
            if a in created:
                consumed.add(a)
            if b in created:
                consumed.add(b)

        killed = created & consumed
        killed_ids = {vocab[t] for t in killed if t in vocab}
        all_ids = set(vocab.values())
        survived_ids = all_ids - killed_ids

        # Count usage on test split
        data_dir = DATA_PATHS[name]
        ds = load_from_disk(f"{data_dir}/{args.split}")
        counts = Counter()
        for row in ds:
            counts.update(row["input_ids"])

        n_killed = max(1, len(killed_ids))
        n_surv = max(1, len(survived_ids))
        V = len(vocab)

        killed_zero = sum(1 for t in killed_ids if counts.get(t, 0) == 0)
        killed_low = sum(1 for t in killed_ids if counts.get(t, 0) <= 10)
        survived_zero = sum(1 for t in survived_ids if counts.get(t, 0) == 0)

        short = SHORT.get(name, name)
        print(
            f"  {short:<20} "
            f"{killed_zero/n_killed:>11.1%} "
            f"{killed_low/n_killed:>16.1%} "
            f"{killed_zero/V:>20.1%} "
            f"{survived_zero/n_surv:>14.1%}"
        )


if __name__ == "__main__":
    main()
