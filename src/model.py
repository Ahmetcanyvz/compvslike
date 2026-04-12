"""Language model module using PyTorch Lightning."""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "me57M-tied"
    model_type: str = "llama"
    hidden_act: str = "silu"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 24
    num_key_value_heads: int = 24
    num_hidden_layers: int = 6
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_llama_config(
        self,
        vocab_size: int,
        max_position_embeddings: int = 2048,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        attn_implementation: str = "sdpa",
    ) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=False,  # Disable KV cache for training
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_implementation,
        )


@dataclass
class OptimConfig:
    """Optimizer and scheduler configuration."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    decay_steps: int = 10000
    min_lr_ratio: float = 0.1
    z_loss_weight: float | None = 1e-4
    weight_decay_embedding: bool = False


def get_attention_implementation() -> str:
    """Determine the best available attention implementation."""
    if importlib.util.find_spec("flash_attn"):
        return "flash_attention_2"
    return "sdpa"


def load_model_from_checkpoint(checkpoint_path: str | Path) -> PreTrainedModel:
    """Load a HuggingFace model from a Lightning checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), weights_only=False, map_location="cpu")

    # Extract model state dict
    state_dict = {
        k.removeprefix("model.").removeprefix("_orig_mod."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }

    # Get config from checkpoint
    config_dict = checkpoint["hyper_parameters"].get("config")
    if isinstance(config_dict, dict):
        config = LlamaConfig(**config_dict)
    else:
        config = config_dict

    # Use eager attention for inference compatibility
    config._attn_implementation = "eager"

    model = LlamaForCausalLM(config)
    model.load_state_dict(state_dict)

    return model


class LanguageModel(LightningModule):
    """Lightning module for causal language modeling with Llama architecture."""

    def __init__(
        self,
        config: dict,
        optim_config: dict | None = None,
        use_flash_attention: bool = True,
        use_liger_kernel: bool = False,
        torch_compile: bool = False,
        max_steps: int = 50000,
    ) -> None:
        super().__init__()

        # Store config as dict for checkpoint serialization
        self.config = config
        self.optim_config = OptimConfig(**(optim_config or {}))
        self.use_flash_attention = use_flash_attention
        self.use_liger_kernel = use_liger_kernel
        self._max_steps = max_steps
        self.torch_compile = torch_compile

        self.save_hyperparameters()

    def configure_model(self) -> None:
        """Initialize the model. Called by Lightning before training."""
        # Determine attention implementation
        if self.use_flash_attention:
            attn_impl = get_attention_implementation()
        else:
            attn_impl = "eager"

        # Build LlamaConfig
        llama_config = LlamaConfig(
            **{k: v for k, v in self.config.items() if k != "name"},
            use_cache=False,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
        )

        # Create model
        self.model = LlamaForCausalLM(llama_config)

        # Apply Liger kernel optimizations
        if self.use_liger_kernel:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama

                torch._dynamo.config.capture_scalar_outputs = True
                apply_liger_kernel_to_llama(
                    rope=True,
                    cross_entropy=False,
                    fused_linear_cross_entropy=True,
                    rms_norm=True,
                    swiglu=True,
                    model=self.model,
                )
            except ImportError:
                print("Warning: liger-kernel not installed, skipping kernel optimizations")

        # Compile model if requested
        if self.torch_compile:
            self.model = torch.compile(self.model)

        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Model initialized: {num_params:.1f}M parameters, attention: {attn_impl}")

    def forward(self, input_ids: Tensor, **kwargs) -> Tensor:
        """Forward pass returning logits."""
        return self.model(input_ids=input_ids, **kwargs).logits

    def _compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute loss with optional z-loss regularization."""
        input_ids = batch["input_ids"]

        outputs = self.model(input_ids=input_ids, labels=input_ids.clone())
        loss = outputs.loss
        logs = {"loss": loss.detach()}

        # Add z-loss regularization if configured
        if self.optim_config.z_loss_weight is not None and self.optim_config.z_loss_weight > 0:
            logits = outputs.logits
            z_loss = logits.logsumexp(dim=-1).pow(2).mean()
            loss = loss + self.optim_config.z_loss_weight * z_loss
            logs["z_loss"] = z_loss.detach()
            logs["total_loss"] = loss.detach()

        # Free logits to avoid OOM during validation with large vocab
        del outputs

        return loss, logs

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        loss, logs = self._compute_loss(batch)

        self.log_dict(
            {f"train/{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step — processes in chunks to avoid OOM with large vocab.

        Liger's fused_linear_cross_entropy doesn't fuse in eval mode,
        so the full logit tensor (batch × seq × vocab) is materialized.
        For 128k vocab this can be 15GB+. Chunking avoids this.
        """
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        chunk_size = max(1, min(4, batch_size))  # process 4 samples at a time

        total_loss = 0.0
        num_chunks = 0

        for i in range(0, batch_size, chunk_size):
            chunk = {"input_ids": input_ids[i:i + chunk_size]}
            chunk_loss, _ = self._compute_loss(chunk)
            total_loss += chunk_loss.detach() * chunk["input_ids"].shape[0]
            num_chunks += chunk["input_ids"].shape[0]

        loss = total_loss / num_chunks
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=batch_size, sync_dist=True)

        return loss

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Test step."""
        loss, logs = self._compute_loss(batch)

        self.log_dict(
            {f"test/{k}": v for k, v in logs.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer with weight decay groups and scheduler."""
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip weight decay for 1D params (biases, norms)
            if param.dim() < 2:
                nodecay_params.append(param)
                continue

            # Optionally skip weight decay for embeddings
            if not self.optim_config.weight_decay_embedding and name.endswith(".weight"):
                module_name = name.rsplit(".weight", 1)[0]
                try:
                    module = self.model.get_submodule(module_name)
                    if isinstance(module, torch.nn.Embedding):
                        nodecay_params.append(param)
                        continue
                except AttributeError:
                    pass

            decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optim_groups,
            lr=self.optim_config.learning_rate,
            betas=(self.optim_config.beta1, self.optim_config.beta2),
            fused=torch.cuda.is_available(),
        )

        # Warmup-stable-decay scheduler
        total_steps = self._max_steps
        warmup_steps = self.optim_config.warmup_steps
        decay_steps = self.optim_config.decay_steps
        stable_steps = max(0, total_steps - warmup_steps - decay_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return current_step / max(1, warmup_steps)
            elif current_step < warmup_steps + stable_steps:
                # Stable phase
                return 1.0
            else:
                # Linear decay
                decay_progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
                return max(self.optim_config.min_lr_ratio, 1.0 - decay_progress * (1.0 - self.optim_config.min_lr_ratio))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
