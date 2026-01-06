"""Training script for language models."""

import os
from pathlib import Path
from typing import Optional

import torch
import typer
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console
from transformers import AutoTokenizer

from src.data import DataModule
from src.model import LanguageModel

app = typer.Typer(help="Train language models with PyTorch Lightning.")
console = Console()


def load_config(config_path: Path) -> dict:
    """Load and validate configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(config_path: Path) -> dict:
    """Load model architecture configuration."""
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    return model_config


class StopAtStepsCallback(Callback):
    """Stop training after a fixed number of optimizer steps."""

    def __init__(self, max_steps: int, grad_accum: int = 1):
        self.max_steps = max_steps
        self.grad_accum = grad_accum
        self.total_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_batches += 1
        step = self.total_batches // self.grad_accum
        if step >= self.max_steps:
            trainer.should_stop = True


class LogToFileCallback(Callback):
    """Log train loss to a text file at regular intervals."""

    def __init__(self, log_path: Path, every_n_steps: int = 1000, grad_accum: int = 1):
        self.log_path = Path(log_path)
        self.every_n_steps = every_n_steps
        self.grad_accum = grad_accum
        self.total_batches = 0
        self.logged_steps = set()
        print(f"[LogToFileCallback] Will write to: {self.log_path}")

    def _write_log(self, msg: str):
        """Write message to log file with flush."""
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()
        print(f"[LOG] {msg}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_batches += 1
        step = self.total_batches // self.grad_accum

        # DEBUG: print when step reaches multiples of 100
        if step > 0 and step % 100 == 0 and step not in self.logged_steps:
            print(f"[DEBUG] step={step}, every_n={self.every_n_steps}, check={step % self.every_n_steps == 0}")

        if step > 0 and step % self.every_n_steps == 0 and step not in self.logged_steps:
            loss = trainer.callback_metrics.get("train/loss", float("nan"))
            self._write_log(f"step={step}, train_loss={float(loss):.4f}")
            self.logged_steps.add(step)

    def on_train_start(self, trainer, pl_module):
        self._write_log("Training log")
        self._write_log("=" * 50)
        self._write_log(f"Logging every {self.every_n_steps} optimizer steps")

    def on_validation_end(self, trainer, pl_module):
        """Log after each validation."""
        step = self.total_batches // self.grad_accum
        loss = trainer.callback_metrics.get("train/loss", float("nan"))
        val_loss = trainer.callback_metrics.get("val/loss", float("nan"))
        self._write_log(f"[VAL] step={step}, train_loss={float(loss):.4f}, val_loss={float(val_loss):.4f}")


class CheckpointAtStepsCallback(Callback):
    """Save checkpoints at specific optimizer steps."""

    def __init__(self, checkpoint_dir: Path, every_n_steps: int, grad_accum: int = 1, save_last: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_steps = every_n_steps
        self.grad_accum = grad_accum
        self.total_batches = 0
        self.save_last = save_last
        self.saved_steps = set()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CheckpointAtStepsCallback] Will save to: {self.checkpoint_dir} every {every_n_steps} steps")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_batches += 1
        step = self.total_batches // self.grad_accum
        if step > 0 and step % self.every_n_steps == 0 and step not in self.saved_steps:
            self.saved_steps.add(step)
            path = self.checkpoint_dir / f"step{step}.ckpt"
            trainer.save_checkpoint(str(path))
            print(f"[CKPT] Saved checkpoint at step {step}: {path}")

    def on_train_end(self, trainer, pl_module):
        """Save final checkpoint."""
        if self.save_last:
            path = self.checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(str(path))
            print(f"[CKPT] Saved final checkpoint: {path}")


def setup_callbacks(config: dict, output_dir: Path, max_steps: int, grad_accum: int) -> list:
    """Set up training callbacks."""
    ckpt_config = config.get("checkpoint", {})
    logging_config = config.get("logging", {})
    checkpoint_dir = output_dir / ckpt_config.get("save_dir", ".checkpoints")
    save_every_n_steps = ckpt_config.get("save_every_n_steps", 5000)
    log_every_n_steps = logging_config.get("log_loss_every_n_steps", 1000)

    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        StopAtStepsCallback(max_steps, grad_accum),
        LogToFileCallback(output_dir / "training_log.txt", every_n_steps=log_every_n_steps, grad_accum=grad_accum),
        CheckpointAtStepsCallback(
            checkpoint_dir=checkpoint_dir,
            every_n_steps=save_every_n_steps,
            grad_accum=grad_accum,
            save_last=ckpt_config.get("save_last", True),
        ),
    ]

    return callbacks


def setup_trainer(config: dict, output_dir: Path) -> Trainer:
    """Set up Lightning Trainer."""
    training_config = config.get("training", {})
    hardware_config = config.get("hardware", {})
    logging_config = config.get("logging", {})

    grad_accum = training_config.get("gradient_accumulation", 1)
    max_steps = training_config.get("max_steps", 50000)

    callbacks = setup_callbacks(config, output_dir, max_steps, grad_accum)
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    trainer = Trainer(
        # Training
        max_steps=max_steps,
        gradient_clip_val=training_config.get("max_grad_norm", 1.0),
        accumulate_grad_batches=grad_accum,
        # Hardware
        accelerator=hardware_config.get("accelerator", "auto"),
        devices=hardware_config.get("devices", "auto"),
        precision=hardware_config.get("precision", "bf16-mixed"),
        strategy=hardware_config.get("strategy", "auto"),
        # Logging
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        val_check_interval=logging_config.get("val_check_interval", 1000),
        # Callbacks
        callbacks=callbacks,
        logger=logger,
        # Misc
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return trainer


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to training config YAML"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
) -> None:
    """Train a language model."""
    # Load configuration
    config = load_config(config_path)
    console.print(f"[green]Loaded config from {config_path}[/green]")

    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed, workers=True)
    console.print(f"[blue]Random seed: {seed}[/blue]")

    # Set up paths
    paths_config = config.get("paths", {})
    training_config = config.get("training", {})

    # Create descriptive output folder name
    base_output_dir = Path(paths_config.get("output_dir", "./outputs"))
    tokenizer_path = paths_config.get("tokenizer", "unknown")
    tokenizer_name = Path(tokenizer_path).name if tokenizer_path else "unknown"
    model_name = config.get("model", {}).get("config_path", "model").split("/")[-1].replace(".yaml", "")
    max_steps = training_config.get("max_steps", 50000)

    run_name = f"{model_name}_{tokenizer_name}_{max_steps}steps"
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[blue]Output dir: {output_dir}[/blue]")

    # Save config to output dir
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load tokenizer to get vocab size
    tokenizer_path = paths_config.get("tokenizer")
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
        eos_token_id = tokenizer.eos_token_id or 0
        console.print(f"[blue]Tokenizer vocab size: {vocab_size}[/blue]")
    else:
        # Default vocab size if no tokenizer provided
        vocab_size = 32000
        eos_token_id = 0
        console.print("[yellow]Warning: No tokenizer path provided, using default vocab_size=32000[/yellow]")

    # Load model config
    model_config_path = Path(config.get("model", {}).get("config_path", "configs/models/me57M-tied.yaml"))
    if not model_config_path.is_absolute():
        model_config_path = config_path.parent / model_config_path

    model_arch_config = load_model_config(model_config_path)
    model_arch_config["vocab_size"] = vocab_size
    model_arch_config["max_position_embeddings"] = config.get("training", {}).get("sequence_length", 2048)

    console.print(f"[blue]Model: {model_arch_config.get('name', 'unknown')}[/blue]")

    # Create DataModule
    data_module = DataModule(
        train_data_path=paths_config.get("train_data"),
        val_data_path=paths_config.get("val_data"),
        test_data_path=paths_config.get("test_data"),
        seq_len=training_config.get("sequence_length", 2048),
        eos_token_id=eos_token_id,
        shuffle_seed=seed,
        batch_size=training_config.get("batch_size", 32),
        eval_batch_size=training_config.get("eval_batch_size"),
        num_workers=config.get("hardware", {}).get("num_workers", 4),
    )

    # Create model
    model_settings = config.get("model", {})
    optim_config = {
        "learning_rate": training_config.get("learning_rate", 3e-4),
        "weight_decay": training_config.get("weight_decay", 0.1),
        "beta1": training_config.get("beta1", 0.9),
        "beta2": training_config.get("beta2", 0.95),
        "warmup_steps": training_config.get("warmup_steps", 2000),
        "decay_steps": training_config.get("decay_steps", 10000),
        "min_lr_ratio": training_config.get("min_lr_ratio", 0.1),
        "z_loss_weight": training_config.get("z_loss_weight"),
    }

    model = LanguageModel(
        config=model_arch_config,
        optim_config=optim_config,
        use_flash_attention=model_settings.get("use_flash_attention", True),
        use_liger_kernel=model_settings.get("use_liger_kernel", False),
        torch_compile=model_settings.get("torch_compile", False),
    )

    # Create trainer
    trainer = setup_trainer(config, output_dir)

    # Train
    console.print("[green]Starting training...[/green]")
    torch.set_float32_matmul_precision("high")

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=str(resume) if resume else None,
    )

    console.print(f"[green]Training complete! Outputs saved to {output_dir}[/green]")


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to training config YAML"),
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint"),
) -> None:
    """Run validation on a trained model."""
    config = load_config(config_path)

    # Set up paths
    paths_config = config.get("paths", {})

    # Load tokenizer
    tokenizer_path = paths_config.get("tokenizer")
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
        eos_token_id = tokenizer.eos_token_id or 0
    else:
        vocab_size = 32000
        eos_token_id = 0

    # Load model config
    model_config_path = Path(config.get("model", {}).get("config_path", "configs/models/me57M-tied.yaml"))
    if not model_config_path.is_absolute():
        model_config_path = config_path.parent / model_config_path

    model_arch_config = load_model_config(model_config_path)
    model_arch_config["vocab_size"] = vocab_size

    # Create DataModule
    training_config = config.get("training", {})
    data_module = DataModule(
        val_data_path=paths_config.get("val_data"),
        seq_len=training_config.get("sequence_length", 2048),
        eos_token_id=eos_token_id,
        batch_size=training_config.get("batch_size", 32),
        num_workers=config.get("hardware", {}).get("num_workers", 4),
    )

    # Load model from checkpoint
    model = LanguageModel.load_from_checkpoint(
        checkpoint,
        config=model_arch_config,
        use_flash_attention=False,  # Use eager for validation
    )

    # Create trainer
    trainer = Trainer(
        accelerator=config.get("hardware", {}).get("accelerator", "auto"),
        devices=1,
        precision=config.get("hardware", {}).get("precision", "bf16-mixed"),
    )

    # Validate
    results = trainer.validate(model, datamodule=data_module)
    console.print(f"[green]Validation results: {results}[/green]")


if __name__ == "__main__":
    app()
