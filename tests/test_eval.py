"""Tests for evaluation and log-probability collection."""

from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from src.eval import batch_by_tokens, collate_with_left_padding, compute_logprobs_sliding_window
from src.model import LanguageModel


class TestBatchByTokens:
    """Tests for dynamic batching by token count."""

    def test_single_batch(self):
        """Test all documents fit in one batch."""
        docs = [{"input_ids": list(range(10))} for _ in range(5)]
        dataset = Dataset.from_list(docs)

        batches = list(batch_by_tokens(dataset, max_tokens_per_batch=100))

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_multiple_batches(self):
        """Test documents split across batches."""
        docs = [{"input_ids": list(range(50))} for _ in range(10)]
        dataset = Dataset.from_list(docs)

        batches = list(batch_by_tokens(dataset, max_tokens_per_batch=150))

        # With 50 tokens each and max 150, should get batches of ~3 docs
        assert len(batches) > 1
        for batch in batches:
            total_tokens = max(len(d["input_ids"]) for d in batch) * len(batch)
            assert total_tokens <= 150 or len(batch) == 1

    def test_variable_length_documents(self):
        """Test batching with variable length documents."""
        docs = [
            {"input_ids": list(range(100))},
            {"input_ids": list(range(50))},
            {"input_ids": list(range(25))},
            {"input_ids": list(range(10))},
        ]
        dataset = Dataset.from_list(docs)

        batches = list(batch_by_tokens(dataset, max_tokens_per_batch=200))

        # All documents should be included
        total_docs = sum(len(b) for b in batches)
        assert total_docs == 4

    def test_empty_dataset(self):
        """Test batching empty dataset."""
        dataset = Dataset.from_list([])
        batches = list(batch_by_tokens(dataset, max_tokens_per_batch=100))
        assert len(batches) == 0


class TestCollateWithLeftPadding:
    """Tests for left-padding collation."""

    def test_uniform_length(self):
        """Test collation with uniform length sequences."""
        batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9, 10]},
        ]

        result = collate_with_left_padding(batch, pad_value=0)

        assert result["input_ids"].shape == (2, 5)
        assert torch.equal(result["input_ids"][0], torch.tensor([1, 2, 3, 4, 5]))

    def test_variable_length(self):
        """Test left-padding for variable length sequences."""
        batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [6, 7, 8]},
        ]

        result = collate_with_left_padding(batch, pad_value=0)

        assert result["input_ids"].shape == (2, 5)
        # Second sequence should be left-padded
        assert torch.equal(result["input_ids"][1], torch.tensor([0, 0, 6, 7, 8]))

    def test_uid_preserved(self):
        """Test that UIDs are preserved."""
        batch = [
            {"input_ids": [1, 2, 3], "uid": 100},
            {"input_ids": [4, 5, 6], "uid": 200},
        ]

        result = collate_with_left_padding(batch, pad_value=0)

        assert result["uid"] == [100, 200]

    def test_custom_pad_value(self):
        """Test custom padding value."""
        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4]},
        ]

        result = collate_with_left_padding(batch, pad_value=-1)

        assert torch.equal(result["input_ids"][1], torch.tensor([-1, -1, 4]))


class TestComputeLogprobs:
    """Tests for log-probability computation."""

    @pytest.fixture
    def simple_model(self, model_57m_config: dict):
        """Create a simple model for testing."""
        model = LanguageModel(
            config=model_57m_config,
            use_flash_attention=False,
        )
        model.configure_model()
        model.eval()
        return model.model

    def test_short_sequence(self, simple_model, vocab_size: int):
        """Test log-prob computation for short sequence."""
        batch_size = 2
        seq_len = 64

        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            log_probs, token_ids = compute_logprobs_sliding_window(
                simple_model,
                input_ids,
                window_size=128,
                step_size=64,
            )

        # Output should have seq_len - 1 positions (no prediction for first token)
        assert log_probs.shape == (batch_size, seq_len - 1)
        assert token_ids.shape == (batch_size, seq_len - 1)

    def test_output_values(self, simple_model, vocab_size: int):
        """Test that log-probs are negative (log of probability)."""
        input_ids = torch.randint(1, vocab_size, (1, 32))

        with torch.no_grad():
            log_probs, _ = compute_logprobs_sliding_window(
                simple_model,
                input_ids,
                window_size=64,
                step_size=32,
            )

        # Log probabilities should be <= 0
        assert torch.all(log_probs <= 0)

    def test_sliding_window(self, simple_model, vocab_size: int):
        """Test sliding window for long sequences."""
        batch_size = 1
        seq_len = 256
        window_size = 64
        step_size = 32

        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            log_probs, token_ids = compute_logprobs_sliding_window(
                simple_model,
                input_ids,
                window_size=window_size,
                step_size=step_size,
            )

        # Should have predictions for all positions except first
        assert log_probs.shape == (batch_size, seq_len - 1)
        assert token_ids.shape == (batch_size, seq_len - 1)

    def test_padding_ignored(self, simple_model, vocab_size: int):
        """Test that padding tokens are handled correctly."""
        # Create sequence with padding at start
        input_ids = torch.randint(1, vocab_size, (1, 32))
        input_ids[0, :5] = 0  # Pad first 5 tokens

        with torch.no_grad():
            log_probs, token_ids = compute_logprobs_sliding_window(
                simple_model,
                input_ids,
                pad_value=0,
            )

        # Padded positions should have 0 log-prob (ignored in cross-entropy)
        # The first 4 positions in output correspond to predicting positions 1-4 (padded)
        assert torch.all(log_probs[0, :4] == 0)

    def test_deterministic(self, simple_model, vocab_size: int):
        """Test that same input gives same output."""
        input_ids = torch.randint(1, vocab_size, (1, 64))

        with torch.no_grad():
            log_probs1, _ = compute_logprobs_sliding_window(simple_model, input_ids)
            log_probs2, _ = compute_logprobs_sliding_window(simple_model, input_ids)

        assert torch.allclose(log_probs1, log_probs2)


class TestEvaluationIntegration:
    """Integration tests for the full evaluation pipeline."""

    def test_full_pipeline(self, model_57m_config: dict, pseudo_dataset: Path, tmp_path: Path):
        """Test running evaluation on a dataset."""
        from datasets import load_from_disk

        # Create model
        model = LanguageModel(
            config=model_57m_config,
            use_flash_attention=False,
        )
        model.configure_model()
        model.eval()

        # Load and process dataset
        dataset = load_from_disk(str(pseudo_dataset))

        # Add length and sort
        dataset = dataset.map(lambda x: {"length": len(x["input_ids"])}).sort("length", reverse=True)

        # Process one batch
        batch_gen = batch_by_tokens(dataset, max_tokens_per_batch=1000)
        batch = next(iter(batch_gen))

        collated = collate_with_left_padding(batch)

        with torch.no_grad():
            log_probs, token_ids = compute_logprobs_sliding_window(
                model.model,
                collated["input_ids"],
                window_size=256,
            )

        # Should get results
        assert log_probs.shape[0] == len(batch)
        assert not torch.isnan(log_probs).any()
