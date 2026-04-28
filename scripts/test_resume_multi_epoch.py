"""Multi-epoch resume test.

Trains past one full epoch to verify that:
1. Resume from mid-epoch-0 checkpoint skips correctly (first iteration)
2. Epoch 1 iterates the full dataset (no re-skip)
3. No data is missed at the epoch boundary

Setup: dataset has 256 indices, 4 ranks × 4 batch_size = 16 indices per step.
One epoch = 256 / 16 = 16 steps. We train for 24 steps to ensure epoch wrap.

Usage:
    export NCCL_NET=Socket
    rm -rf /tmp/test_epoch
    torchrun --nproc_per_node=4 scripts/test_resume_multi_epoch.py --phase clean
    torchrun --nproc_per_node=4 scripts/test_resume_multi_epoch.py --phase resume
    python scripts/test_resume_multi_epoch.py --phase verify
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer, LightningDataModule
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/tmp/test_epoch")
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "ckpts"
DATASET_SIZE = 256
BATCH_SIZE = 4
STEPS_CLEAN = 4   # mid-epoch-0
STEPS_FINAL = 24  # past epoch 0 (16 steps) into epoch 1


class IndexDataset(Dataset):
    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, idx: int) -> dict:
        return {"idx": torch.tensor(idx, dtype=torch.long)}


class SkipBatchSampler(torch.utils.data.BatchSampler):
    _DEBUG_RANK = 0

    def __init__(self, sampler, batch_size, drop_last=True, skip_batches: int = 0):
        super().__init__(sampler, batch_size, drop_last)
        self.skip_batches = skip_batches
        self._consumed = False
        self._iter_count = 0

    def __iter__(self):
        skip = 0 if self._consumed else self.skip_batches
        self._iter_count += 1
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == self._DEBUG_RANK:
            print(f"[DEBUG iter#{self._iter_count}] _consumed={self._consumed}, skip={skip}, super_len={super().__len__()}", flush=True)
        self._consumed = True
        yielded = 0
        for i, batch in enumerate(super().__iter__()):
            if i >= skip:
                yield batch
                yielded += 1
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == self._DEBUG_RANK:
            print(f"[DEBUG iter#{self._iter_count}] yielded {yielded} batches (out of {super().__len__()} super, skip={skip})", flush=True)

    def __len__(self):
        # Always return total epoch length; Lightning subtracts batch_progress itself.
        result = super().__len__()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == self._DEBUG_RANK:
            print(f"[DEBUG __len__] _consumed={self._consumed}, returning {result}", flush=True)
        return result


class TestDataModule(LightningDataModule):
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


def make_trainer(max_steps: int) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(ROOT),
        use_distributed_sampler=False,
    )


def write_log(rank: int, phase: str, indices: list[int]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{phase}_rank{rank}.json").write_text(json.dumps(indices))


def run_clean() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    dm = TestDataModule()
    trainer = make_trainer(STEPS_CLEAN)
    trainer.fit(model, datamodule=dm)
    rank = trainer.global_rank
    write_log(rank, "clean", model.seen_indices)
    ckpt = CKPT_DIR / "mid_epoch.ckpt"
    trainer.save_checkpoint(str(ckpt))
    if rank == 0:
        print(f"[rank0] saved {ckpt}")


def run_baseline() -> None:
    """Train from scratch all the way to STEPS_FINAL — reference for comparison."""
    model = TinyModel()
    dm = TestDataModule()
    trainer = make_trainer(STEPS_FINAL)
    trainer.fit(model, datamodule=dm)
    rank = trainer.global_rank
    write_log(rank, "baseline", model.seen_indices)


def run_resume() -> None:
    ckpt = CKPT_DIR / "mid_epoch.ckpt"
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
    trainer = make_trainer(STEPS_FINAL)
    trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))
    rank = trainer.global_rank
    write_log(rank, "resume", model.seen_indices)


def verify() -> None:
    print("\n=== Multi-epoch verification ===\n")
    clean_all, resume_all = [], []
    for r in range(4):
        c = json.loads((LOG_DIR / f"clean_rank{r}.json").read_text())
        rs = json.loads((LOG_DIR / f"resume_rank{r}.json").read_text())
        clean_all.append(c)
        resume_all.append(rs)

    # Each rank in clean saw 4 batches × 4 = 16 indices (epoch 0, batches 0-3)
    # Each rank in resume should see:
    #   - skip first 4 batches (already seen)
    #   - 12 batches of remaining epoch 0 (batches 4-15) = 48 indices
    #   - then epoch 1 starts: 8 more batches × 4 = 32 indices (batches 0-7)
    # Total per rank in resume: 12*4 + 8*4 = 80 indices
    # Steps: 24 total - 4 already done = 20 batches per rank? No wait.
    # max_steps is per-trainer, so total steps = STEPS_FINAL = 24, and we trained 4 already.
    # So resume runs for 24-4 = 20 steps. Each step = 1 batch per rank.
    # Per rank: 20 batches = 80 indices.

    for r in range(4):
        print(f"rank{r} clean ({len(clean_all[r])} idx):  {clean_all[r]}")
        print(f"rank{r} resume ({len(resume_all[r])} idx): {resume_all[r]}")
        print()

    # Per-rank check 1: resume continues from where clean left off
    # Per rank with shuffle=False: rank R sees [R, R+4, R+8, ..., R+60] in clean (16 indices)
    # Then in resume continues with [R+64, R+68, ..., R+252] (epoch 0 remainder, 48 indices)
    # Then epoch 1 starts: [R, R+4, ...] (32 more indices)
    # So resume should: NOT overlap with clean within epoch-0 remainder, then re-see clean indices in epoch 1
    print("--- Per-rank epoch-0 continuity ---")
    epoch_0_per_rank = (DATASET_SIZE // 4) // BATCH_SIZE  # batches per rank per epoch
    skipped = STEPS_CLEAN  # batches already done
    expected_remaining = epoch_0_per_rank - skipped  # 16 - 4 = 12 batches
    expected_e0_idx = expected_remaining * BATCH_SIZE  # 48 indices
    print(f"Expected epoch 0 remainder per rank: {expected_remaining} batches = {expected_e0_idx} indices")
    print()

    all_ok = True
    for r in range(4):
        first_resume = resume_all[r][0]
        last_clean = clean_all[r][-1]
        expected_first = last_clean + 4  # next index in rank-R's stride
        if first_resume != expected_first:
            print(f"❌ rank{r}: first_resume={first_resume}, expected={expected_first} — discontinuity at epoch-0 boundary")
            all_ok = False
        else:
            print(f"✅ rank{r}: epoch 0 continues at {first_resume} (was at {last_clean})")
    print()

    # Check that epoch 1 is iterated (resume contains indices < 64, which are epoch 1 re-visits)
    print("--- Epoch 1 wrap check (should re-see early indices) ---")
    for r in range(4):
        # In epoch 0, rank R saw indices [R, R+4, ..., R+(rank_indices*4 - 4)]
        # At epoch 1 start, rank R sees [R, R+4, ...] again
        # The resume's last 32 indices should be epoch 1 batches 0-7 → [R, R+4, ..., R+28]
        epoch1_start = resume_all[r][expected_e0_idx]  # first index of epoch 1 in resume log
        if epoch1_start == r:
            print(f"✅ rank{r}: epoch 1 starts at index {epoch1_start} (correct)")
        else:
            print(f"❌ rank{r}: epoch 1 starts at index {epoch1_start}, expected {r}")
            all_ok = False
    print()

    # CRITICAL CHECK: clean+resume must equal baseline (from-scratch run).
    # This validates that resume sees EXACTLY the same data as if no resume happened.
    print("--- Baseline equivalence check ---")
    baseline_all = []
    try:
        for r in range(4):
            b = json.loads((LOG_DIR / f"baseline_rank{r}.json").read_text())
            baseline_all.append(b)
    except FileNotFoundError:
        print("(no baseline run found — skip --phase baseline to compare)")
        baseline_all = None

    if baseline_all is not None:
        baseline_ok = True
        for r in range(4):
            combined = clean_all[r] + resume_all[r]
            if combined == baseline_all[r]:
                print(f"✅ rank{r}: clean+resume matches baseline ({len(combined)} indices)")
            else:
                # Find first divergence
                for i, (c, b) in enumerate(zip(combined, baseline_all[r])):
                    if c != b:
                        print(f"❌ rank{r}: divergence at index {i}: combined={c}, baseline={b}")
                        break
                if len(combined) != len(baseline_all[r]):
                    print(f"❌ rank{r}: length mismatch combined={len(combined)} baseline={len(baseline_all[r])}")
                baseline_ok = False
        all_ok = all_ok and baseline_ok

    print()
    if all_ok:
        print("✅✅ PASS — multi-epoch resume works correctly")
    else:
        print("❌ FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["clean", "resume", "baseline", "verify"], required=True)
    args = parser.parse_args()
    if args.phase == "clean":
        run_clean()
    elif args.phase == "resume":
        run_resume()
    elif args.phase == "baseline":
        run_baseline()
    else:
        verify()


if __name__ == "__main__":
    main()
