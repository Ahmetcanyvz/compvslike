"""Integration tests for training."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from lightning.pytorch import Trainer

from src.data import DataModule
from src.model import LanguageModel


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""

    @pytest.fixture
    def training_config(self, tmp_path: Path, pseudo_dataset: Path, vocab_size: int) -> Path:
        """Create a training config file."""
        config = {
            "paths": {
                "train_data": str(pseudo_dataset),
                "val_data": str(pseudo_dataset),
                "output_dir": str(tmp_path / "outputs"),
            },
            "model": {
                "config_path": "configs/models/me57M-tied.yaml",
                "use_flash_attention": False,
                "use_liger_kernel": False,
                "torch_compile": False,
            },
            "training": {
                "seed": 42,
                "max_steps": 2,
                "batch_size": 2,
                "sequence_length": 64,
                "learning_rate": 1e-4,
                "weight_decay": 0.1,
                "warmup_steps": 1,
                "decay_steps": 1,
                "gradient_accumulation": 1,
            },
            "hardware": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": 32,
            },
            "logging": {
                "log_every_n_steps": 1,
                "val_check_interval": 1,
            },
            "checkpoint": {
                "save_every_n_steps": 1,
                "save_last": True,
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    def test_training_two_steps(self, model_57m_config: dict, pseudo_dataset: Path, seq_len: int):
        """Test that training runs for two steps without errors."""
        # Create DataModule
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,  # Shorter for faster test
            batch_size=2,
            num_workers=0,
        )

        # Create model with smaller config for testing
        model = LanguageModel(
            config=model_57m_config,
            optim_config={
                "learning_rate": 1e-4,
                "warmup_steps": 1,
                "decay_steps": 1,
            },
            use_flash_attention=False,
        )

        # Create trainer
        trainer = Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            precision=32,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Train
        trainer.fit(model, datamodule=dm)

        # Verify training completed
        assert trainer.global_step == 2

    def test_training_with_validation(self, model_57m_config: dict, pseudo_dataset: Path):
        """Test training with validation checks."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,
            batch_size=2,
            num_workers=0,
        )

        model = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-4, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        trainer = Trainer(
            max_steps=2,
            val_check_interval=1,
            accelerator="cpu",
            devices=1,
            precision=32,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=dm)

        # Check validation was run
        assert trainer.callback_metrics.get("val/loss") is not None

    def test_checkpoint_saving(self, model_57m_config: dict, pseudo_dataset: Path, tmp_path: Path):
        """Test that checkpoints are saved correctly."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,
            batch_size=2,
            num_workers=0,
        )

        model = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-4, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        from lightning.pytorch.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            every_n_train_steps=1,
            save_last=True,
        )

        trainer = Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            precision=32,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=dm)

        # Check checkpoint was saved
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.ckpt"))
        assert len(checkpoint_files) > 0

    def test_resume_from_checkpoint(self, model_57m_config: dict, pseudo_dataset: Path, tmp_path: Path):
        """Test resuming training from checkpoint."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,
            batch_size=2,
            num_workers=0,
        )

        model = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-4, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        from lightning.pytorch.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            every_n_train_steps=1,
            save_last=True,
        )

        # Initial training
        trainer1 = Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            precision=32,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer1.fit(model, datamodule=dm)
        checkpoint_path = tmp_path / "checkpoints" / "last.ckpt"

        # Resume training
        model2 = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-4, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        trainer2 = Trainer(
            max_steps=4,
            accelerator="cpu",
            devices=1,
            precision=32,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer2.fit(model2, datamodule=dm, ckpt_path=str(checkpoint_path))

        # Check resumed from step 2 and continued to step 4
        assert trainer2.global_step == 4

    def test_gradient_accumulation(self, model_57m_config: dict, pseudo_dataset: Path):
        """Test training with gradient accumulation."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,
            batch_size=1,
            num_workers=0,
        )

        model = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-4, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        trainer = Trainer(
            max_steps=2,
            accumulate_grad_batches=2,  # Accumulate over 2 batches
            accelerator="cpu",
            devices=1,
            precision=32,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=dm)

        assert trainer.global_step == 2

    def test_loss_decreases(self, model_57m_config: dict, pseudo_dataset: Path):
        """Test that loss generally decreases during training (or at least doesn't explode)."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=64,
            batch_size=4,
            num_workers=0,
        )

        model = LanguageModel(
            config=model_57m_config,
            optim_config={"learning_rate": 1e-3, "warmup_steps": 1, "decay_steps": 1},
            use_flash_attention=False,
        )

        # Track losses
        losses = []

        class LossTracker:
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                losses.append(outputs.item())

        trainer = Trainer(
            max_steps=10,
            accelerator="cpu",
            devices=1,
            precision=32,
            callbacks=[LossTracker()],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(model, datamodule=dm)

        # Loss should not explode (stay within reasonable bounds)
        assert all(loss < 100 for loss in losses), f"Loss exploded: {losses}"
        # Loss should not be NaN
        assert all(not torch.isnan(torch.tensor(loss)) for loss in losses)
