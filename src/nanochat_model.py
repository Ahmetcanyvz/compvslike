"""Lightning module wrapping nanochat's GPT model and MuonAdamW optimizer.

Uses nanochat's model architecture and optimizer while keeping
lm-trainer's data loading and training infrastructure.
"""

import math
import sys
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

# Add nanochat to path
NANOCHAT_DIR = Path(__file__).parent.parent / "nanochat"
if str(NANOCHAT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_DIR))

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW


class NanochatLanguageModel(LightningModule):
    """Lightning module using nanochat's GPT architecture and MuonAdamW optimizer."""

    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 12,
        n_embd: int = 768,
        n_head: int = 6,
        n_kv_head: int = 6,
        sequence_len: int = 2048,
        window_pattern: str = "SSSL",
        # Optimizer hyperparams
        embedding_lr: float = 0.2,
        unembedding_lr: float = 0.004,
        matrix_lr: float = 0.02,
        scalar_lr: float = 0.5,
        weight_decay: float = 0.28,
        # Schedule
        warmup_steps: int = 40,
        warmdown_ratio: float = 0.65,
        final_lr_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gpt_config = GPTConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            window_pattern=window_pattern,
        )

        # Store optimizer hyperparams
        self.embedding_lr = embedding_lr
        self.unembedding_lr = unembedding_lr
        self.matrix_lr = matrix_lr
        self.scalar_lr = scalar_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmdown_ratio = warmdown_ratio
        self.final_lr_frac = final_lr_frac

        # Automatic optimization off — we handle optimizer.step() manually
        # because MuonAdamW needs per-step momentum/WD schedule updates
        self.automatic_optimization = False

    def configure_model(self) -> None:
        """Initialize model. Called by Lightning before training."""
        self.model = GPT(self.gpt_config)
        self.model.init_weights()

        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"nanochat GPT initialized: {num_params:.1f}M parameters")

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        """Forward pass."""
        return self.model(input_ids, targets=targets)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Manual training step with MuonAdamW schedule updates."""
        input_ids = batch["input_ids"]
        optimizer = self.optimizers()
        scheduler_info = self._get_schedule_info()

        # Update optimizer hyperparams per step
        lrm = self._get_lr_multiplier(self.global_step)
        muon_momentum = self._get_muon_momentum(self.global_step)
        muon_wd = self._get_weight_decay(self.global_step)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group.get("kind") == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        # Forward + loss (model returns loss directly when targets given)
        loss = self.model(input_ids[:, :-1], targets=input_ids[:, 1:])

        # Backward
        self.manual_backward(loss)

        # Step optimizer every accumulate_grad_batches
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.log_dict(
            {"train/loss": loss.detach(), "train/lr": lrm * self.matrix_lr},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=input_ids.shape[0],
        )

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        input_ids = batch["input_ids"]
        loss = self.model(input_ids[:, :-1], targets=input_ids[:, 1:])
        self.log("val/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=input_ids.shape[0], sync_dist=True)
        return loss

    def configure_optimizers(self) -> Any:
        """Set up MuonAdamW optimizer with nanochat's param groups."""
        return self.model.setup_optimizer(
            unembedding_lr=self.unembedding_lr,
            embedding_lr=self.embedding_lr,
            matrix_lr=self.matrix_lr,
            weight_decay=self.weight_decay,
            scalar_lr=self.scalar_lr,
        )

    def _get_total_steps(self) -> int:
        return int(self.trainer.estimated_stepping_batches)

    def _get_lr_multiplier(self, step: int) -> float:
        total = self._get_total_steps()
        warmdown_iters = round(self.warmdown_ratio * total)

        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        elif step <= total - warmdown_iters:
            return 1.0
        else:
            progress = (total - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * self.final_lr_frac

    def _get_muon_momentum(self, step: int) -> float:
        total = self._get_total_steps()
        warmdown_iters = round(self.warmdown_ratio * total)
        warmdown_start = total - warmdown_iters

        if step < 400:
            frac = step / 400
            return (1 - frac) * 0.85 + frac * 0.97
        elif step >= warmdown_start:
            progress = (step - warmdown_start) / warmdown_iters
            return 0.97 * (1 - progress) + 0.90 * progress
        else:
            return 0.97

    def _get_weight_decay(self, step: int) -> float:
        total = self._get_total_steps()
        # Scale weight decay by batch size ratio (nanochat default)
        return self.weight_decay * 0.5 * (1 + math.cos(math.pi * step / total))

    def _get_schedule_info(self) -> dict:
        return {
            "total_steps": self._get_total_steps(),
            "warmup_steps": self.warmup_steps,
            "warmdown_ratio": self.warmdown_ratio,
        }
