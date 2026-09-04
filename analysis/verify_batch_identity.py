"""Prove the batched estimator selection is identical to per-document selection.

The tokenizer returns the same token count for a document whether it is encoded
alone or inside a batch (no padding in the returned id lists, no cross-document
interaction). Given that, this checks the ONLY other thing that could differ:
the collection loop's ordering, filtering, and stop boundary.

We mirror both the per-doc loop and the batched loop as pure functions and assert
they produce identical (accepted docs, order, uids, total_tokens, break point)
across many random inputs and batch sizes.

Run:
    python scripts/verify_batch_identity.py
"""

import random


def per_doc(counts, target, min_tokens, uid_offset=0):
    docs, total = [], 0
    for i, n in enumerate(counts):
        if n < min_tokens:
            continue
        docs.append((i, uid_offset + len(docs)))  # (stream_index, uid)
        total += n
        if total >= target:
            break
    return docs, total


def batched(counts, target, min_tokens, batch_size, uid_offset=0):
    docs, total = [], 0
    done = False
    pos = 0
    while not done and pos < len(counts):
        batch = counts[pos:pos + batch_size]
        pos += len(batch)
        for j, n in enumerate(batch):
            idx = pos - len(batch) + j
            if n < min_tokens:
                continue
            docs.append((idx, uid_offset + len(docs)))
            total += n
            if total >= target:
                done = True
                break
    return docs, total


def main():
    rng = random.Random(0)
    trials = 20000
    for t in range(trials):
        n_docs = rng.randint(0, 400)
        counts = [rng.randint(0, 300) for _ in range(n_docs)]
        target = rng.randint(1, 20000)
        min_tokens = rng.choice([0, 50, 100])
        bs = rng.choice([1, 2, 7, 100, 1000])

        a = per_doc(counts, target, min_tokens)
        b = batched(counts, target, min_tokens, bs)

        if a != b:
            print("MISMATCH")
            print(" counts:", counts)
            print(" target:", target, "min:", min_tokens, "batch:", bs)
            print(" per_doc:", a)
            print(" batched:", b)
            raise SystemExit(1)

    print(f"OK: {trials:,} random trials — batched selection is identical to per-doc")
    print("    (same accepted docs, same order, same uids, same total, same stop boundary)")


if __name__ == "__main__":
    main()
