"""Tests for the language model module."""

import pytest
import torch
from unittest.mock import MagicMock

from src.model import LanguageModel, ModelConfig, OptimConfig, get_attention_implementation


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        """Test default model config values."""
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 6
        assert config.tie_word_embeddings is True

    def test_to_llama_config(self):
        """Test conversion to LlamaConfig."""
        config = ModelConfig()
        llama_config = config.to_llama_config(vocab_size=1000)

        assert llama_config.vocab_size == 1000
        assert llama_config.hidden_size == 768
        assert llama_config.num_hidden_layers == 6


class TestOptimConfig:
    """Tests for optimizer configuration."""

    def test_default_config(self):
        """Test default optimizer config."""
        config = OptimConfig()
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.1
        assert config.warmup_steps == 2000

    def test_custom_config(self):
        """Test custom optimizer config."""
        config = OptimConfig(learning_rate=1e-3, warmup_steps=100)
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 100


class TestLanguageModel:
    """Tests for LanguageModel LightningModule."""

    def test_model_creation(self, model_57m_config: dict, optim_config: dict):
        """Test model can be created."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        assert model is not None
        assert model.config == model_57m_config

    def test_configure_model(self, model_57m_config: dict, optim_config: dict):
        """Test configure_model initializes the underlying model."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        assert hasattr(model, "model")
        assert model.model is not None

        # Check model has correct architecture
        num_params = sum(p.numel() for p in model.model.parameters())
        assert num_params > 0

    def test_forward_pass(self, model_57m_config: dict, sample_batch: dict, seq_len: int):
        """Test forward pass produces correct output shape."""
        model = LanguageModel(
            config=model_57m_config,
            use_flash_attention=False,
        )
        model.configure_model()
        model.eval()

        with torch.no_grad():
            logits = model(sample_batch["input_ids"])

        batch_size = sample_batch["input_ids"].shape[0]
        vocab_size = model_57m_config["vocab_size"]

        assert logits.shape == (batch_size, seq_len + 1, vocab_size)

    def test_training_step(self, model_57m_config: dict, optim_config: dict, sample_batch: dict):
        """Test training step computes loss."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()
        model.train()

        # Mock the log_dict method
        model.log_dict = MagicMock()

        loss = model.training_step(sample_batch, batch_idx=0)

        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss > 0  # Loss should be positive

    def test_validation_step(self, model_57m_config: dict, optim_config: dict, sample_batch: dict):
        """Test validation step computes loss."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()
        model.eval()

        # Mock the log_dict method
        model.log_dict = MagicMock()

        with torch.no_grad():
            loss = model.validation_step(sample_batch, batch_idx=0)

        assert loss is not None
        assert loss > 0

    def test_z_loss(self, model_57m_config: dict, sample_batch: dict):
        """Test z-loss regularization is applied."""
        optim_config = {"z_loss_weight": 1e-4}

        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        # Mock log_dict to capture what's logged
        logged_values = {}

        def mock_log_dict(d, **kwargs):
            logged_values.update(d)

        model.log_dict = mock_log_dict

        model.training_step(sample_batch, batch_idx=0)

        # Z-loss should be logged
        assert "train/z_loss" in logged_values
        assert "train/total_loss" in logged_values

    def test_no_z_loss(self, model_57m_config: dict, sample_batch: dict):
        """Test z-loss can be disabled."""
        optim_config = {"z_loss_weight": None}

        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        logged_values = {}
        model.log_dict = lambda d, **kwargs: logged_values.update(d)

        model.training_step(sample_batch, batch_idx=0)

        # Z-loss should NOT be logged
        assert "train/z_loss" not in logged_values

    def test_configure_optimizers(self, model_57m_config: dict, optim_config: dict):
        """Test optimizer configuration."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        model.trainer = mock_trainer

        opt_dict = model.configure_optimizers()

        assert "optimizer" in opt_dict
        assert "lr_scheduler" in opt_dict

        optimizer = opt_dict["optimizer"]
        assert len(optimizer.param_groups) == 2  # decay and no-decay groups

    def test_weight_decay_groups(self, model_57m_config: dict, optim_config: dict):
        """Test that weight decay is applied correctly to param groups."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 1000
        model.trainer = mock_trainer

        opt_dict = model.configure_optimizers()
        optimizer = opt_dict["optimizer"]

        # Check param groups have different weight decay
        weight_decays = [g["weight_decay"] for g in optimizer.param_groups]
        assert weight_decays[0] > 0  # Decay group
        assert weight_decays[1] == 0  # No-decay group

    def test_lr_scheduler(self, model_57m_config: dict, optim_config: dict):
        """Test learning rate scheduler warmup-stable-decay."""
        model = LanguageModel(
            config=model_57m_config,
            optim_config=optim_config,
            use_flash_attention=False,
        )
        model.configure_model()

        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 100
        model.trainer = mock_trainer

        opt_dict = model.configure_optimizers()
        scheduler = opt_dict["lr_scheduler"]["scheduler"]

        # Get learning rates at different steps
        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase during warmup
        assert lrs[0] < lrs[optim_config["warmup_steps"] - 1]

        # LR should decrease during decay
        assert lrs[-1] < lrs[optim_config["warmup_steps"]]

    def test_model_with_tied_embeddings(self, model_57m_config: dict):
        """Test model with tied embeddings has shared weights."""
        model_57m_config["tie_word_embeddings"] = True

        model = LanguageModel(config=model_57m_config, use_flash_attention=False)
        model.configure_model()

        # Check embedding and output weights are the same
        embed_weight = model.model.model.embed_tokens.weight
        output_weight = model.model.lm_head.weight

        assert torch.equal(embed_weight, output_weight)

    def test_model_with_untied_embeddings(self, model_57m_config: dict):
        """Test model with untied embeddings has separate weights."""
        model_57m_config["tie_word_embeddings"] = False

        model = LanguageModel(config=model_57m_config, use_flash_attention=False)
        model.configure_model()

        # Count total parameters - should be more with untied
        num_params = sum(p.numel() for p in model.model.parameters())

        # Create tied version for comparison
        model_57m_config["tie_word_embeddings"] = True
        model_tied = LanguageModel(config=model_57m_config, use_flash_attention=False)
        model_tied.configure_model()
        num_params_tied = sum(p.numel() for p in model_tied.model.parameters())

        assert num_params > num_params_tied


class TestAttentionImplementation:
    """Tests for attention implementation selection."""

    def test_get_attention_implementation(self):
        """Test attention implementation returns valid string."""
        impl = get_attention_implementation()
        assert impl in ["flash_attention_2", "sdpa", "eager"]
