"""Test DDP resume correctness.

Trains a tiny model for 8 steps with a deterministic dataset where each "sample"
is just its index. Each rank logs the set of indices it sees. We then resume
from a checkpoint at step 4 and continue to step 8. The test PASSES iff the
post-resume run sees indices that pick up where the pre-resume run left off,
not indices from the start of the dataset.

Usage on clariden (interactive 4xGH200 node):
    srun --partition=normal --account=a139 --time=00:30:00 --nodes=1 \\
        --ntasks-per-node=1 --cpus-per-task=72 --gpus-per-node=4 \\
        --container-writable --environment=lm_trainer_env --pty bash

    # inside the allocation:
    cd /iopsstor/scratch/cscs/ayavuz/compvslike
    pip install -e . --no-deps
    rm -rf /tmp/test_resume

    # Phase 1: train fresh, checkpoint at step 4, stop at 4
    torchrun --nproc_per_node=4 scripts/test_resume.py --phase fresh

    # Phase 2: resume from step 4 ckpt, train to step 8
    torchrun --nproc_per_node=4 scripts/test_resume.py --phase resume

    # Phase 3: print verdict
    python scripts/test_resume.py --phase verify
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/tmp/test_resume")
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "ckpts"
DATASET_SIZE = 256
BATCH_SIZE = 4  # per rank; 4 ranks × bs 4 = 16 indices per step
STEPS_PHASE1 = 4
STEPS_PHASE2 = 8


class IndexDataset(Dataset):
    """Each sample IS its global index — easy to track what each rank consumed."""

    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, idx: int) -> dict:
        return {"idx": torch.tensor(idx, dtype=torch.long)}


class SkipBatchSampler(torch.utils.data.BatchSampler):
    """Same as in src/data.py — needed to test the actual resume path."""

    def __init__(self, sampler, batch_size, drop_last=True, skip_batches: int = 0):
        super().__init__(sampler, batch_size, drop_last)
        self.skip_batches = skip_batches

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            if i >= self.skip_batches:
                yield batch

    def __len__(self):
        return max(0, super().__len__() - self.skip_batches)


class TinyModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.seen_indices: list[int] = []

    def training_step(self, batch, batch_idx):
        idxs = batch["idx"].tolist()
        self.seen_indices.extend(idxs)
        # bogus loss that depends on params so optimizer has work
        x = batch["idx"].float().unsqueeze(-1)
        return self.layer(x).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def make_dataloader(skip: int = 0) -> DataLoader:
    ds = IndexDataset()
    if skip > 0:
        sampler = torch.utils.data.SequentialSampler(ds)
        bs = SkipBatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=True, skip_batches=skip)
        return DataLoader(ds, batch_sampler=bs, num_workers=0)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)


def write_log(rank: int, phase: str, indices: list[int]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{phase}_rank{rank}.json"
    out.write_text(json.dumps(indices))


def run_fresh() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    trainer = Trainer(
        max_steps=STEPS_PHASE1,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(ROOT),
    )
    trainer.fit(model, train_dataloaders=make_dataloader(skip=0))
    rank = trainer.global_rank
    write_log(rank, "fresh", model.seen_indices)
    # save_checkpoint is a collective — must be called on all ranks
    ckpt = CKPT_DIR / "step4.ckpt"
    trainer.save_checkpoint(str(ckpt))
    if rank == 0:
        print(f"[rank0] saved {ckpt}")


def run_resume() -> None:
    ckpt = CKPT_DIR / "step4.ckpt"
    assert ckpt.exists(), f"missing {ckpt} — run --phase fresh first"

    # Read batch_completed from checkpoint, set skip
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
    trainer = Trainer(
        max_steps=STEPS_PHASE2,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(ROOT),
    )
    trainer.fit(
        model,
        train_dataloaders=make_dataloader(skip=batch_completed),
        ckpt_path=str(ckpt),
    )
    rank = trainer.global_rank
    write_log(rank, "resume", model.seen_indices)


def verify() -> None:
    print("\n=== Verification ===\n")
    fresh_all, resume_all = [], []
    for r in range(4):
        f = json.loads((LOG_DIR / f"fresh_rank{r}.json").read_text())
        rs = json.loads((LOG_DIR / f"resume_rank{r}.json").read_text())
        fresh_all.append(f)
        resume_all.append(rs)
        print(f"rank{r} fresh:  {f}")
        print(f"rank{r} resume: {rs}")
        print()

    # Check 1: ranks see disjoint data within each phase (DDP sharding works)
    print("--- DDP sharding check (each rank should see different data) ---")
    for phase_name, phase_data in [("fresh", fresh_all), ("resume", resume_all)]:
        rank_sets = [set(r) for r in phase_data]
        cross_rank_overlap = set()
        for i in range(len(rank_sets)):
            for j in range(i + 1, len(rank_sets)):
                cross_rank_overlap |= rank_sets[i] & rank_sets[j]
        if cross_rank_overlap:
            print(f"❌ {phase_name}: ranks share indices {sorted(cross_rank_overlap)} — DDP NOT sharding")
        else:
            print(f"✅ {phase_name}: all ranks saw disjoint data")
    print()

    # Check 2: resume doesn't re-feed pre-resume data
    fresh_set = set(i for r in fresh_all for i in r)
    resume_set = set(i for r in resume_all for i in r)
    overlap = fresh_set & resume_set

    print("--- Resume continuity check ---")
    print(f"fresh saw indices:  {sorted(fresh_set)}")
    print(f"resume saw indices: {sorted(resume_set)}")
    print(f"overlap:            {sorted(overlap)}")
    print()

    if not overlap:
        print("✅ PASS — resume continued without re-feeding data")
    else:
        print(f"❌ FAIL — resume re-fed {len(overlap)} indices the model already saw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["fresh", "resume", "verify"], required=True)
    args = parser.parse_args()
    if args.phase == "fresh":
        run_fresh()
    elif args.phase == "resume":
        run_resume()
    else:
        verify()


if __name__ == "__main__":
    main()
