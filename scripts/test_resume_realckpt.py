"""End-to-end resume test using a REAL checkpoint, real 1B model, real data.

Phases:
  baseline: load real_ckpt, train 200 steps → log fingerprints[0:200] (no save)
  clean:    load real_ckpt, train 100 steps → save step+100.ckpt, log fingerprints[0:100]
  resume:   load step+100.ckpt, train 100 more steps → log fingerprints[100:200]
  verify:   compare baseline[100:200] == resume[100:200]

Each batch's "fingerprint" is sum(first 8 tokens). Order-sensitive across ranks.

Usage (interactive node, 4xGH200):
    export NCCL_NET=Socket
    cd /iopsstor/scratch/cscs/ayavuz/compvslike
    rm -rf /tmp/test_realckpt

    REAL_CKPT=/iopsstor/scratch/cscs/ayavuz/compvslike/outputs/me1B-tied_greedyll-exact-128k_20Btok_seed42/.checkpoints/step20000.ckpt
    DATA=/iopsstor/scratch/cscs/ayavuz/compvslike/data/fineweb-edu-greedyll-exact-128k
    TOK=/iopsstor/scratch/cscs/ayavuz/compvslike/tokenizers/greedyll-exact-128k

    torchrun --nproc_per_node=4 scripts/test_resume_realckpt.py --phase baseline --real-ckpt $REAL_CKPT --data $DATA --tok $TOK
    torchrun --nproc_per_node=4 scripts/test_resume_realckpt.py --phase clean    --real-ckpt $REAL_CKPT --data $DATA --tok $TOK
    torchrun --nproc_per_node=4 scripts/test_resume_realckpt.py --phase resume   --real-ckpt $REAL_CKPT --data $DATA --tok $TOK
    python scripts/test_resume_realckpt.py --phase verify
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from lightning.pytorch import Callback, Trainer
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from src.data import PackedTokenDataset, SkipBatchSampler
from src.model import LanguageModel, ModelConfig, OptimConfig

ROOT = Path("/tmp/test_realckpt")
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "ckpts"

STEPS_BASELINE_TOTAL = 200  # trained beyond clean
STEPS_CLEAN_DELTA = 100      # how many steps clean trains
STEPS_RESUME_TOTAL = 200     # resume target = clean + 100


class FingerprintCallback(Callback):
    """Records the sum of first 8 tokens in every training batch (per-rank)."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.fingerprints: list[int] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        ids = batch["input_ids"]
        for row in ids:
            self.fingerprints.append(int(row[:8].sum().item()))

    def on_fit_end(self, trainer, pl_module) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rank = trainer.global_rank
        out = self.log_path.with_name(f"{self.log_path.stem}.rank{rank}.json")
        out.write_text(json.dumps(self.fingerprints))


def build_dataloader(data_path: Path, batch_size: int, seq_len: int, skip: int) -> DataLoader:
    ds = PackedTokenDataset(
        data_path, seq_len=seq_len, eos_token_id=0, shuffle_seed=42,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = torch.utils.data.SequentialSampler(ds)
    bs = SkipBatchSampler(sampler, batch_size=batch_size, drop_last=True, skip_batches=skip)
    return DataLoader(ds, batch_sampler=bs, num_workers=4, pin_memory=True)


def build_model(tokenizer_path: Path) -> LanguageModel:
    """Build same architecture as the real run (me1B-tied)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config = AutoConfig.from_pretrained(
        Path(__file__).parent.parent / "models" / "me1B-tied.yaml",
    )
    # Fall back: just use whatever the real ckpt was trained with
    return None  # Built from checkpoint instead


def get_batch_completed(ckpt_path: Path) -> int:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return (
        state.get("loops", {})
        .get("fit_loop", {})
        .get("epoch_loop.batch_progress", {})
        .get("total", {})
        .get("completed", 0)
    )


def get_global_step(ckpt_path: Path) -> int:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return state.get("global_step", 0)


def run_phase(phase: str, real_ckpt: Path, data: Path, tok: Path) -> None:
    """Run a phase of the test. Loads REAL ckpt, trains delta steps, optionally saves."""
    if phase == "baseline":
        load_ckpt = real_ckpt
        delta_steps = STEPS_BASELINE_TOTAL
        save_after = False
    elif phase == "clean":
        load_ckpt = real_ckpt
        delta_steps = STEPS_CLEAN_DELTA
        save_after = True
    elif phase == "resume":
        load_ckpt = CKPT_DIR / "clean.ckpt"
        delta_steps = STEPS_RESUME_TOTAL - STEPS_CLEAN_DELTA  # 100 more
        save_after = False
    else:
        raise ValueError(phase)

    if not load_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_ckpt}")

    base_step = get_global_step(load_ckpt)
    batch_completed = get_batch_completed(load_ckpt)
    target_max_steps = base_step + delta_steps

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f"[{phase}] Loading {load_ckpt.name}: global_step={base_step}, batch_completed={batch_completed}", flush=True)
        print(f"[{phase}] Will train to max_steps={target_max_steps} (delta={delta_steps})", flush=True)

    # Load model from checkpoint to get exact architecture
    model = LanguageModel.load_from_checkpoint(load_ckpt, strict=False, max_steps=target_max_steps)

    # Build dataloader with proper skip
    dataloader = build_dataloader(data, batch_size=16, seq_len=2048, skip=batch_completed)

    # Fingerprint logger
    fp_log = LOG_DIR / f"{phase}_fp"
    fp_callback = FingerprintCallback(fp_log)

    # Checkpoint saver only for clean phase
    callbacks = [fp_callback]
    if save_after:
        from lightning.pytorch.callbacks import ModelCheckpoint
        ckpt_callback = ModelCheckpoint(
            dirpath=str(CKPT_DIR),
            filename="clean",
            save_top_k=1,
            every_n_train_steps=delta_steps,
            save_last=False,
        )
        callbacks.append(ckpt_callback)

    trainer = Trainer(
        max_steps=target_max_steps,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision="bf16-true",
        enable_checkpointing=save_after,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(ROOT),
        use_distributed_sampler=False,
        callbacks=callbacks,
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=str(load_ckpt))


def verify() -> None:
    print("\n=== Real-checkpoint resume verification ===\n")
    all_ok = True
    for r in range(4):
        baseline = json.loads((LOG_DIR / f"baseline_fp.rank{r}.json").read_text())
        clean = json.loads((LOG_DIR / f"clean_fp.rank{r}.json").read_text())
        resume = json.loads((LOG_DIR / f"resume_fp.rank{r}.json").read_text())

        n_clean = len(clean)
        baseline_post_clean = baseline[n_clean:]

        if clean == baseline[:n_clean]:
            print(f"✅ rank{r}: clean[0:{n_clean}] == baseline[0:{n_clean}]")
        else:
            print(f"❌ rank{r}: clean diverges from baseline")
            all_ok = False
            for i, (c, b) in enumerate(zip(clean, baseline)):
                if c != b:
                    print(f"   first divergence at {i}: clean={c}, baseline={b}")
                    break

        if resume == baseline_post_clean:
            print(f"✅ rank{r}: resume == baseline[{n_clean}:{len(baseline)}] ({len(resume)} fingerprints)")
        else:
            print(f"❌ rank{r}: resume diverges from baseline tail")
            print(f"   baseline_post_clean[:5] = {baseline_post_clean[:5]}")
            print(f"   resume[:5]              = {resume[:5]}")
            all_ok = False
            for i, (b, rs) in enumerate(zip(baseline_post_clean, resume)):
                if b != rs:
                    print(f"   first divergence at {i}: baseline={b}, resume={rs}")
                    break
            if len(resume) != len(baseline_post_clean):
                print(f"   length mismatch: resume={len(resume)}, baseline_post_clean={len(baseline_post_clean)}")

    print()
    if all_ok:
        print("✅✅ PASS — real checkpoint resume is data-identical to baseline")
    else:
        print("❌ FAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "clean", "resume", "verify"], required=True)
    parser.add_argument("--real-ckpt", type=Path, help="Path to real step{N}.ckpt")
    parser.add_argument("--data", type=Path, help="Path to data dir (containing train/val/test)")
    parser.add_argument("--tok", type=Path, help="Path to tokenizer dir")
    args = parser.parse_args()

    if args.phase == "verify":
        verify()
    else:
        if not (args.real_ckpt and args.data and args.tok):
            raise SystemExit("--real-ckpt, --data, --tok all required for non-verify phases")
        run_phase(args.phase, args.real_ckpt, args.data / "train", args.tok)


if __name__ == "__main__":
    main()
