"""Pytest fixtures for lm-trainer tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset


@pytest.fixture
def vocab_size() -> int:
    """Vocabulary size for tests (kept small for speed)."""
    return 1000


@pytest.fixture
def seq_len() -> int:
    """Sequence length for tests."""
    return 128


@pytest.fixture
def pseudo_dataset(vocab_size: int, seq_len: int, tmp_path: Path) -> Path:
    """Create a pseudo tokenized dataset with random tokens.

    Creates 100 variable-length documents for testing on-the-fly packing.
    """
    rng = np.random.default_rng(42)

    # Create variable-length documents (50-200 tokens each)
    documents = []
    for i in range(100):
        length = rng.integers(50, 200)
        input_ids = rng.integers(1, vocab_size, size=length).tolist()
        documents.append({"input_ids": input_ids, "uid": i})

    dataset = Dataset.from_list(documents)
    data_path = tmp_path / "pseudo_dataset"
    dataset.save_to_disk(str(data_path))

    return data_path


@pytest.fixture
def packed_dataset(vocab_size: int, seq_len: int, tmp_path: Path) -> Path:
    """Create a pre-packed dataset with fixed-length sequences."""
    rng = np.random.default_rng(42)

    # Create fixed-length sequences
    sequences = []
    for i in range(50):
        input_ids = rng.integers(1, vocab_size, size=seq_len + 1).tolist()
        sequences.append({"input_ids": input_ids})

    dataset = Dataset.from_list(sequences)
    data_path = tmp_path / "packed_dataset"
    dataset.save_to_disk(str(data_path))

    return data_path


@pytest.fixture
def model_57m_config(vocab_size: int) -> dict:
    """Real 57M model architecture config.

    This uses the actual 57M architecture but with a smaller vocab for faster tests.
    """
    return {
        "name": "me57M-tied",
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_act": "silu",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 24,
        "num_key_value_heads": 24,
        "num_hidden_layers": 6,
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
    }


@pytest.fixture
def optim_config() -> dict:
    """Optimizer configuration for tests."""
    return {
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "warmup_steps": 10,
        "decay_steps": 10,
        "min_lr_ratio": 0.1,
        "z_loss_weight": 1e-4,
    }


@pytest.fixture
def sample_batch(vocab_size: int, seq_len: int) -> dict[str, torch.Tensor]:
    """Create a sample batch for testing."""
    rng = np.random.default_rng(42)
    batch_size = 4
    input_ids = rng.integers(1, vocab_size, size=(batch_size, seq_len + 1))
    return {"input_ids": torch.from_numpy(input_ids).long()}
