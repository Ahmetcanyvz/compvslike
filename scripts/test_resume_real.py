"""End-to-end test: resume from a clean checkpoint with the actual src/data.py
and src/train.py fix path, verifying resume continuity in DDP.

Phase 1 (clean): trains 0→4 steps, saves step4.ckpt (the "clean" checkpoint).
Phase 2 (resume from clean): loads step4.ckpt, trains 4→8 steps.
Phase 3 (verify): asserts resume saw indices that come AFTER fresh's, no overlap.

Uses the same DataLoader construction pattern as src/data.py:
- Trainer(use_distributed_sampler=False)
- Manual DistributedSampler(shuffle=False)
- SkipBatchSampler(skip_batches=batch_completed)

Usage:
    export NCCL_NET=Socket
    rm -rf /tmp/test_real
    torchrun --nproc_per_node=4 scripts/test_resume_real.py --phase clean
    torchrun --nproc_per_node=4 scripts/test_resume_real.py --phase resume
    python scripts/test_resume_real.py --phase verify
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer, LightningDataModule
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/tmp/test_real")
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "ckpts"
DATASET_SIZE = 256
BATCH_SIZE = 4
STEPS_CLEAN = 4
STEPS_RESUME = 8


class IndexDataset(Dataset):
    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, idx: int) -> dict:
        return {"idx": torch.tensor(idx, dtype=torch.long)}


class SkipBatchSampler(torch.utils.data.BatchSampler):
    """Same impl as src/data.py — skip applies only on first iteration."""

    def __init__(self, sampler, batch_size, drop_last=True, skip_batches: int = 0):
        super().__init__(sampler, batch_size, drop_last)
        self.skip_batches = skip_batches
        self._consumed = False

    def __iter__(self):
        skip = 0 if self._consumed else self.skip_batches
        self._consumed = True
        for i, batch in enumerate(super().__iter__()):
            if i >= skip:
                yield batch

    def __len__(self):
        return super().__len__()


class TestDataModule(LightningDataModule):
    """Mirror of src/data.py train_dataloader logic exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.train_ds = IndexDataset()
        self._skip_batches = 0

    def train_dataloader(self) -> DataLoader:
        skip = self._skip_batches

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_ds,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(self.train_ds)

        bs = SkipBatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=True, skip_batches=skip)
        if skip > 0:
            print(f"[rank{torch.distributed.get_rank()}] Dataloader: skipping {skip} batches, {len(bs)} remaining")
        return DataLoader(self.train_ds, batch_sampler=bs, num_workers=0)


class TinyModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.seen_indices: list[int] = []

    def training_step(self, batch, batch_idx):
        self.seen_indices.extend(batch["idx"].tolist())
        x = batch["idx"].float().unsqueeze(-1)
        return self.layer(x).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def write_log(rank: int, phase: str, indices: list[int]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{phase}_rank{rank}.json").write_text(json.dumps(indices))


def make_trainer(max_steps: int, ckpt_enable: bool = False) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        enable_checkpointing=ckpt_enable,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(ROOT),
        use_distributed_sampler=False,
    )


def run_clean() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    dm = TestDataModule()
    trainer = make_trainer(STEPS_CLEAN)
    trainer.fit(model, datamodule=dm)
    rank = trainer.global_rank
    write_log(rank, "clean", model.seen_indices)
    ckpt = CKPT_DIR / "step4_clean.ckpt"
    trainer.save_checkpoint(str(ckpt))
    if rank == 0:
        print(f"[rank0] saved CLEAN checkpoint: {ckpt}")


def run_resume() -> None:
    ckpt = CKPT_DIR / "step4_clean.ckpt"
    assert ckpt.exists()

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    batch_completed = (
        state.get("loops", {})
        .get("fit_loop", {})
        .get("epoch_loop.batch_progress", {})
        .get("total", {})
        .get("completed", 0)
    )
    print(f"[ckpt] batch_completed = {batch_completed}")

    model = TinyModel()
    dm = TestDataModule()
    dm._skip_batches = batch_completed

    trainer = make_trainer(STEPS_RESUME)
    trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))
    rank = trainer.global_rank
    write_log(rank, "resume", model.seen_indices)


def verify() -> None:
    print("\n=== Verification ===\n")
    clean_all, resume_all = [], []
    for r in range(4):
        c = json.loads((LOG_DIR / f"clean_rank{r}.json").read_text())
        rs = json.loads((LOG_DIR / f"resume_rank{r}.json").read_text())
        clean_all.append(c)
        resume_all.append(rs)
        print(f"rank{r} clean:  {c}")
        print(f"rank{r} resume: {rs}")
        print()

    # DDP sharding check
    print("--- DDP sharding check ---")
    for name, data in [("clean", clean_all), ("resume", resume_all)]:
        rank_sets = [set(r) for r in data]
        overlap = set()
        for i in range(len(rank_sets)):
            for j in range(i + 1, len(rank_sets)):
                overlap |= rank_sets[i] & rank_sets[j]
        if overlap:
            print(f"❌ {name}: ranks share {sorted(overlap)} — DDP NOT sharding")
        else:
            print(f"✅ {name}: all ranks saw disjoint data")
    print()

    # Resume continuity check
    clean_set = set(i for r in clean_all for i in r)
    resume_set = set(i for r in resume_all for i in r)
    overlap = clean_set & resume_set

    print("--- Resume continuity check ---")
    print(f"clean indices:  {sorted(clean_set)}")
    print(f"resume indices: {sorted(resume_set)}")
    print(f"overlap:        {sorted(overlap)}")
    print()

    if not overlap:
        # Per-rank continuity: resume should pick up where clean left off
        per_rank_ok = True
        for r in range(4):
            last_clean = clean_all[r][-1]
            first_resume = resume_all[r][0]
            # In our setup with shuffle=False and 4 ranks, rank R sees [R, R+4, R+8, ...].
            # After 4 batches (16 indices) of size 4 per rank, last clean = R + 60, first resume = R + 64.
            expected = last_clean + 4
            ok = first_resume == expected
            print(f"rank{r}: last_clean={last_clean}, first_resume={first_resume}, expected={expected} {'✅' if ok else '❌'}")
            per_rank_ok = per_rank_ok and ok
        print()
        if per_rank_ok:
            print("✅✅ PASS — resume is CORRECT and CONTINUOUS")
        else:
            print("⚠️  no overlap but resume didn't continue exactly where clean left off")
    else:
        print(f"❌ FAIL — resume re-fed {len(overlap)} indices")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["clean", "resume", "verify"], required=True)
    args = parser.parse_args()
    if args.phase == "clean":
        run_clean()
    elif args.phase == "resume":
        run_resume()
    else:
        verify()


if __name__ == "__main__":
    main()
