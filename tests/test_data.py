"""Tests for data loading and packing."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import DataConfig, DataModule, OffsetLocator, PackedTokenDataset, SimpleTokenDataset


class TestOffsetLocator:
    """Tests for OffsetLocator binary search."""

    def test_locate_single(self):
        """Test locating a single position."""
        offsets = np.array([0, 10, 25, 50, 100])
        locator = OffsetLocator(offsets, block_size=2)

        assert locator.locate(0) == 0
        assert locator.locate(5) == 0
        assert locator.locate(10) == 1
        assert locator.locate(15) == 1
        assert locator.locate(25) == 2
        assert locator.locate(49) == 2
        assert locator.locate(50) == 3
        assert locator.locate(99) == 3

    def test_locate_batch(self):
        """Test vectorized batch location."""
        offsets = np.array([0, 10, 25, 50, 100])
        locator = OffsetLocator(offsets, block_size=2)

        positions = np.array([0, 15, 50, 99])
        result = locator.locate_batch(positions)

        expected = np.array([0, 1, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_total_tokens(self):
        """Test total tokens property."""
        offsets = np.array([0, 100, 250, 500])
        locator = OffsetLocator(offsets)
        assert locator.total_tokens == 500

    def test_get_single(self):
        """Test getting single offset."""
        offsets = np.array([0, 10, 25, 50])
        locator = OffsetLocator(offsets)

        assert locator.get(0) == 0
        assert locator.get(1) == 10
        assert locator.get(3) == 50

    def test_get_out_of_bounds(self):
        """Test out of bounds access."""
        offsets = np.array([0, 10, 25])
        locator = OffsetLocator(offsets)

        with pytest.raises(IndexError):
            locator.get(5)


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_valid_config(self, tmp_path: Path):
        """Test valid configuration."""
        config = DataConfig(
            train_data_path=tmp_path,
            seq_len=2048,
            batch_size=32,
        )
        assert config.seq_len == 2048
        assert config.batch_size == 32

    def test_path_conversion(self, tmp_path: Path):
        """Test path conversion from string."""
        config = DataConfig(train_data_path=str(tmp_path))
        assert isinstance(config.train_data_path, Path)

    def test_effective_eval_batch_size(self):
        """Test eval batch size defaults to batch_size."""
        config = DataConfig(batch_size=32)
        assert config.effective_eval_batch_size == 32

        config = DataConfig(batch_size=32, eval_batch_size=16)
        assert config.effective_eval_batch_size == 16


class TestPackedTokenDataset:
    """Tests for on-the-fly packing dataset."""

    def test_dataset_creation(self, pseudo_dataset: Path, seq_len: int):
        """Test dataset initialization."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            eos_token_id=0,
            shuffle_seed=42,
        )

        assert len(dataset) > 0
        assert dataset.seq_len == seq_len + 1  # +1 for next-token prediction

    def test_getitem_returns_correct_shape(self, pseudo_dataset: Path, seq_len: int):
        """Test that __getitem__ returns correct shape."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            eos_token_id=0,
        )

        item = dataset[0]
        assert "input_ids" in item
        assert item["input_ids"].shape == (seq_len + 1,)

    def test_getitem_returns_tensors(self, pseudo_dataset: Path, seq_len: int):
        """Test that __getitem__ returns torch tensors."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            eos_token_id=0,
        )

        item = dataset[0]
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_shuffle_produces_different_order(self, pseudo_dataset: Path, seq_len: int):
        """Test that different shuffle seeds produce different orderings."""
        dataset1 = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            shuffle_seed=42,
        )
        dataset2 = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            shuffle_seed=123,
        )

        # Different seeds should produce different sequence indices
        assert not np.array_equal(dataset1.seq_idx[:10], dataset2.seq_idx[:10])

    def test_no_shuffle(self, pseudo_dataset: Path, seq_len: int):
        """Test dataset without shuffling."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            shuffle_seed=None,
        )

        # Without shuffling, doc_idx should be ordered
        assert np.array_equal(dataset.doc_idx[:10], np.arange(10))

    def test_out_of_bounds_index(self, pseudo_dataset: Path, seq_len: int):
        """Test out of bounds index raises error."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
        )

        with pytest.raises(IndexError):
            dataset[len(dataset) + 1]

    def test_metadata_files_created(self, pseudo_dataset: Path, seq_len: int):
        """Test that metadata files are created."""
        dataset = PackedTokenDataset(
            data_path=pseudo_dataset,
            seq_len=seq_len,
            shuffle_seed=42,
        )

        # Check metadata files exist
        metadata_files = list(pseudo_dataset.glob("*.npy"))
        assert len(metadata_files) >= 2  # At least docs and offsets


class TestSimpleTokenDataset:
    """Tests for pre-packed dataset."""

    def test_dataset_creation(self, packed_dataset: Path):
        """Test simple dataset initialization."""
        dataset = SimpleTokenDataset(data_path=packed_dataset)
        assert len(dataset) == 50

    def test_getitem(self, packed_dataset: Path, seq_len: int):
        """Test __getitem__ returns correct data."""
        dataset = SimpleTokenDataset(data_path=packed_dataset)
        item = dataset[0]

        assert "input_ids" in item
        assert item["input_ids"].shape == (seq_len + 1,)

    def test_shuffle(self, packed_dataset: Path):
        """Test shuffling works."""
        dataset1 = SimpleTokenDataset(data_path=packed_dataset, shuffle_seed=42)
        dataset2 = SimpleTokenDataset(data_path=packed_dataset, shuffle_seed=123)

        # Different seeds should give different first items
        item1 = dataset1[0]["input_ids"]
        item2 = dataset2[0]["input_ids"]
        assert not torch.equal(item1, item2)


class TestDataModule:
    """Tests for Lightning DataModule."""

    def test_datamodule_creation(self, pseudo_dataset: Path, seq_len: int):
        """Test DataModule initialization."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=seq_len,
            batch_size=4,
        )

        assert dm.train_data_path == pseudo_dataset
        assert dm.batch_size == 4

    def test_setup_creates_datasets(self, pseudo_dataset: Path, seq_len: int):
        """Test setup creates train and val datasets."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=seq_len,
            batch_size=4,
        )

        dm.setup(stage="fit")

        assert dm.train_ds is not None
        assert dm.val_ds is not None

    def test_train_dataloader(self, pseudo_dataset: Path, seq_len: int):
        """Test train dataloader works."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            seq_len=seq_len,
            batch_size=4,
            num_workers=0,
        )
        dm.setup(stage="fit")

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 4  # batch size
        assert batch["input_ids"].shape[1] == seq_len + 1  # seq len

    def test_val_dataloader(self, pseudo_dataset: Path, seq_len: int):
        """Test validation dataloader works."""
        dm = DataModule(
            train_data_path=pseudo_dataset,
            val_data_path=pseudo_dataset,
            seq_len=seq_len,
            batch_size=4,
            num_workers=0,
        )
        dm.setup(stage="fit")

        loader = dm.val_dataloader()
        batch = next(iter(loader))

        assert "input_ids" in batch

    def test_use_packing_false(self, packed_dataset: Path, seq_len: int):
        """Test DataModule with pre-packed data."""
        dm = DataModule(
            train_data_path=packed_dataset,
            seq_len=seq_len,
            batch_size=4,
            use_packing=False,
            num_workers=0,
        )
        dm.setup(stage="fit")

        assert isinstance(dm.train_ds, SimpleTokenDataset)
