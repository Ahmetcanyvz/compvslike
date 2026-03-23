"""Training script using nanochat's GPT model with lm-trainer's data pipeline.

Usage:
    uv run python -m src.train_nanochat train <config.yaml> [--seed 42]
"""

from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from transformers import AutoTokenizer

from src.data import DataModule
from src.nanochat_model import NanochatLanguageModel
from src.train import (
    CheckpointAtStepsCallback,
    LogToFileCallback,
    StopAtStepsCallback,
)

app = typer.Typer(help="Train nanochat GPT models with lm-trainer data pipeline.")
console = Console()


# Map from lm-trainer model names to nanochat GPT dimensions
MODEL_CONFIGS = {
    "nc57M": dict(n_layer=6, n_embd=768, n_head=12, n_kv_head=12),
    "nc100M": dict(n_layer=12, n_embd=768, n_head=12, n_kv_head=12),
    "nc340M": dict(n_layer=20, n_embd=1280, n_head=20, n_kv_head=4),
    "nc500M": dict(n_layer=26, n_embd=1280, n_head=20, n_kv_head=4),
    "nc1B": dict(n_layer=22, n_embd=2048, n_head=32, n_kv_head=4),
}


def compute_max_steps(training_config: dict) -> int:
    """Compute max_steps from config."""
    max_tokens = training_config.get("max_tokens")
    max_steps = training_config.get("max_steps")

    if max_tokens is not None:
        batch_size = training_config.get("batch_size", 8)
        grad_accum = training_config.get("gradient_accumulation", 1)
        seq_len = training_config.get("sequence_length", 2048)
        tokens_per_step = batch_size * grad_accum * seq_len
        max_steps = max_tokens // tokens_per_step
        console.print(f"[blue]max_tokens={max_tokens:,} / {tokens_per_step:,} = {max_steps:,} steps[/blue]")
    elif max_steps is None:
        max_steps = 50000

    return max_steps


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to training config YAML"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override seed"),
) -> None:
    """Train a nanochat GPT model."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Seed
    if seed is None:
        seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed, workers=True)
    console.print(f"[blue]Seed: {seed}[/blue]")

    paths_config = config.get("paths", {})
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    hardware_config = config.get("hardware", {})
    logging_config = config.get("logging", {})

    # Load tokenizer for vocab size
    tokenizer_path = paths_config.get("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    eos_token_id = tokenizer.eos_token_id or 0
    console.print(f"[blue]Tokenizer: {tokenizer_path} (vocab={vocab_size:,})[/blue]")

    # Get model dimensions
    nc_model_name = model_config.get("nanochat_model", "nc100M")
    if nc_model_name in MODEL_CONFIGS:
        model_dims = MODEL_CONFIGS[nc_model_name]
    else:
        raise ValueError(f"Unknown nanochat model: {nc_model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")

    console.print(f"[blue]Model: {nc_model_name} (layers={model_dims['n_layer']}, dim={model_dims['n_embd']})[/blue]")

    # Compute steps
    max_steps = compute_max_steps(training_config)
    training_config["max_steps"] = max_steps

    # Output directory
    base_output_dir = Path(paths_config.get("output_dir", "./outputs"))
    tokenizer_name = Path(tokenizer_path).name if tokenizer_path else "unknown"
    max_tokens = training_config.get("max_tokens")
    if max_tokens is not None:
        token_label = f"{max_tokens // 1_000_000_000}B" if max_tokens >= 1_000_000_000 else f"{max_tokens // 1_000_000}M"
        run_name = f"{nc_model_name}_{tokenizer_name}_{token_label}tok_seed{seed}"
    else:
        run_name = f"{nc_model_name}_{tokenizer_name}_{max_steps}steps_seed{seed}"

    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[blue]Output: {output_dir}[/blue]")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create data module
    data_module = DataModule(
        train_data_path=paths_config.get("train_data"),
        val_data_path=paths_config.get("val_data"),
        test_data_path=paths_config.get("test_data"),
        seq_len=training_config.get("sequence_length", 2048),
        eos_token_id=eos_token_id,
        shuffle_seed=seed,
        batch_size=training_config.get("batch_size", 32),
        eval_batch_size=training_config.get("eval_batch_size"),
        num_workers=hardware_config.get("num_workers", 4),
    )

    # Create model
    optim_config = training_config.get("optimizer", {})
    model = NanochatLanguageModel(
        vocab_size=vocab_size,
        n_layer=model_dims["n_layer"],
        n_embd=model_dims["n_embd"],
        n_head=model_dims["n_head"],
        n_kv_head=model_dims["n_kv_head"],
        sequence_len=training_config.get("sequence_length", 2048),
        window_pattern=model_config.get("window_pattern", "SSSL"),
        embedding_lr=optim_config.get("embedding_lr", 0.2),
        unembedding_lr=optim_config.get("unembedding_lr", 0.004),
        matrix_lr=optim_config.get("matrix_lr", 0.02),
        scalar_lr=optim_config.get("scalar_lr", 0.5),
        weight_decay=optim_config.get("weight_decay", 0.28),
        warmup_steps=optim_config.get("warmup_steps", 40),
        warmdown_ratio=optim_config.get("warmdown_ratio", 0.65),
        final_lr_frac=optim_config.get("final_lr_frac", 0.05),
    )

    # Callbacks
    grad_accum = training_config.get("gradient_accumulation", 1)
    ckpt_config = config.get("checkpoint", {})
    checkpoint_dir = output_dir / ckpt_config.get("save_dir", ".checkpoints")
    save_every = ckpt_config.get("save_every_n_steps", 5000)
    log_every = logging_config.get("log_loss_every_n_steps", 1000)

    callbacks = [
        RichProgressBar(),
        StopAtStepsCallback(max_steps),
        LogToFileCallback(output_dir / "training_log.txt", grad_accum=grad_accum, every_n_steps=log_every),
        CheckpointAtStepsCallback(
            checkpoint_dir=checkpoint_dir,
            grad_accum=grad_accum,
            every_n_steps=save_every,
            save_last=ckpt_config.get("save_last", True),
        ),
    ]

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    # Trainer
    trainer = Trainer(
        max_steps=max_steps,
        # gradient_clip_val not supported with manual optimization
        accumulate_grad_batches=grad_accum,
        accelerator=hardware_config.get("accelerator", "auto"),
        devices=hardware_config.get("devices", "auto"),
        precision=hardware_config.get("precision", "bf16-true"),
        strategy=hardware_config.get("strategy", "auto"),
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        val_check_interval=logging_config.get("val_check_interval", 0.5),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    console.print("[green]Starting nanochat training...[/green]")
    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, datamodule=data_module)
    console.print(f"[green]Training complete! Output: {output_dir}[/green]")


if __name__ == "__main__":
    app()
