"""End-to-end resume test using the REAL PackedTokenDataset (memmap shuffle).

Creates a tiny synthetic Arrow dataset, runs PackedTokenDataset on it (which
generates deterministic memmap-based shuffle files), then trains:
1. CLEAN: 0 → 4 steps, save checkpoint
2. RESUME: resume from step 4 ckpt → 24 steps
3. BASELINE: from-scratch 0 → 24 steps
4. VERIFY: clean+resume == baseline (data sequence identity)

Usage:
    export NCCL_NET=Socket
    rm -rf /tmp/test_packed
    torchrun --nproc_per_node=4 scripts/test_resume_packed.py --phase setup
    torchrun --nproc_per_node=4 scripts/test_resume_packed.py --phase clean
    torchrun --nproc_per_node=4 scripts/test_resume_packed.py --phase resume
    torchrun --nproc_per_node=4 scripts/test_resume_packed.py --phase baseline
    python scripts/test_resume_packed.py --phase verify
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from lightning.pytorch import LightningModule, Trainer, LightningDataModule
from torch.utils.data import DataLoader

from src.data import PackedTokenDataset, SkipBatchSampler

ROOT = Path("/tmp/test_packed")
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "ckpts"
DATA_DIR = ROOT / "data"
SEQ_LEN = 16  # short to make many sequences with small data
EOS_TOKEN = 0
SHUFFLE_SEED = 42
BATCH_SIZE = 4  # per-rank
STEPS_CLEAN = 4
STEPS_FINAL = 24


def setup_data() -> None:
    """Create a small synthetic Arrow dataset and pre-create memmap files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_docs = 200
    # Variable-length docs of 10-30 tokens, vocab_size=1000
    examples = {
        "input_ids": [
            rng.integers(1, 1000, size=int(rng.integers(10, 30))).tolist()
            for _ in range(n_docs)
        ]
    }
    ds = Dataset.from_dict(examples)
    save_path = DATA_DIR / "train"
    ds.save_to_disk(str(save_path))
    # Trigger PackedTokenDataset to create memmap shuffle files
    ptd = PackedTokenDataset(save_path, seq_len=SEQ_LEN, eos_token_id=EOS_TOKEN, shuffle_seed=SHUFFLE_SEED)
    print(f"Created dataset: {n_docs} docs → {ptd.num_sequences} packed sequences")
    print(f"Memmap files: {list(save_path.glob('*.npy'))}")


class PackedDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_ds: PackedTokenDataset | None = None
        self._skip_batches = 0

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = PackedTokenDataset(
            DATA_DIR / "train",
            seq_len=SEQ_LEN,
            eos_token_id=EOS_TOKEN,
            shuffle_seed=SHUFFLE_SEED,
        )

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
    """Tracks token-level fingerprint of every batch — order-sensitive."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.fingerprints: list[int] = []

    def training_step(self, batch, batch_idx):
        # Hash each row's first 8 tokens — fingerprint of WHAT data was seen
        ids = batch["input_ids"]
        for row in ids:
            self.fingerprints.append(int(row[:8].sum().item()))
        return self.layer(ids[:, :1].float()).mean()

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


def write_log(rank: int, phase: str, fingerprints: list[int]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{phase}_rank{rank}.json").write_text(json.dumps(fingerprints))


def run_clean() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model = TinyModel()
    dm = PackedDataModule()
    trainer = make_trainer(STEPS_CLEAN)
    trainer.fit(model, datamodule=dm)
    write_log(trainer.global_rank, "clean", model.fingerprints)
    ckpt = CKPT_DIR / "clean.ckpt"
    trainer.save_checkpoint(str(ckpt))


def run_resume() -> None:
    ckpt = CKPT_DIR / "clean.ckpt"
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    batch_completed = (
        state.get("loops", {})
        .get("fit_loop", {})
        .get("epoch_loop.batch_progress", {})
        .get("total", {})
        .get("completed", 0)
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[ckpt] batch_completed = {batch_completed}")
    model = TinyModel()
    dm = PackedDataModule()
    dm._skip_batches = batch_completed
    trainer = make_trainer(STEPS_FINAL)
    trainer.fit(model, datamodule=dm, ckpt_path=str(ckpt))
    write_log(trainer.global_rank, "resume", model.fingerprints)


def run_baseline() -> None:
    model = TinyModel()
    dm = PackedDataModule()
    trainer = make_trainer(STEPS_FINAL)
    trainer.fit(model, datamodule=dm)
    write_log(trainer.global_rank, "baseline", model.fingerprints)


def verify() -> None:
    print("\n=== PackedTokenDataset resume verification ===\n")
    all_ok = True
    for r in range(4):
        clean = json.loads((LOG_DIR / f"clean_rank{r}.json").read_text())
        resume = json.loads((LOG_DIR / f"resume_rank{r}.json").read_text())
        baseline = json.loads((LOG_DIR / f"baseline_rank{r}.json").read_text())

        combined = clean + resume
        if combined == baseline:
            print(f"✅ rank{r}: clean+resume == baseline ({len(combined)} fingerprints)")
        else:
            all_ok = False
            print(f"❌ rank{r}: divergence")
            print(f"   clean ({len(clean)}):    {clean[:8]}...")
            print(f"   resume ({len(resume)}):  {resume[:8]}...")
            print(f"   combined ({len(combined)}): {combined[:8]}...")
            print(f"   baseline ({len(baseline)}): {baseline[:8]}...")
            for i, (c, b) in enumerate(zip(combined, baseline)):
                if c != b:
                    print(f"   first divergence at i={i}: combined={c}, baseline={b}")
                    break
            if len(combined) != len(baseline):
                print(f"   length mismatch: combined={len(combined)}, baseline={len(baseline)}")

    print()
    if all_ok:
        print("✅✅ PASS — PackedTokenDataset resume is data-identical to baseline")
    else:
        print("❌ FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["setup", "clean", "resume", "baseline", "verify"], required=True)
    args = parser.parse_args()
    if args.phase == "setup":
        setup_data()
    elif args.phase == "clean":
        run_clean()
    elif args.phase == "resume":
        run_resume()
    elif args.phase == "baseline":
        run_baseline()
    else:
        verify()


if __name__ == "__main__":
    main()
