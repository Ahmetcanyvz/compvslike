"""Data loading and packing for language model training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset, load_from_disk
from lightning.pytorch import LightningDataModule
from pydantic import BaseModel, field_validator
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler


class DataConfig(BaseModel):
    """Configuration for data loading."""

    train_data_path: Path | None = None
    val_data_path: Path | None = None
    test_data_path: Path | None = None
    seq_len: int = 2048
    eos_token_id: int = 0
    shuffle_seed: int | None = 42
    batch_size: int = 32
    eval_batch_size: int | None = None
    num_workers: int = 4
    pin_memory: bool = True

    @field_validator("train_data_path", "val_data_path", "test_data_path", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        if v is None:
            return None
        return Path(v)

    @property
    def effective_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size


class OffsetLocator:
    """Efficient binary search for document positions using 2-level indexing.

    This enables O(log n) lookup of which document a given token position belongs to,
    using a blocked array structure for cache efficiency.
    """

    def __init__(self, offsets: np.ndarray, block_size: int = 2048) -> None:
        if offsets.ndim != 1:
            raise ValueError(f"Expected offsets to be a 1D array, but got {offsets.ndim}D array.")

        self.offsets = offsets
        self.block_size = block_size
        self.total_len = len(offsets)

        # Pad offsets to fit blocks evenly
        pad_len = (block_size - self.total_len % block_size) % block_size
        if pad_len > 0:
            offsets = np.concatenate([offsets, np.full(pad_len, offsets[-1] + 1)])

        self.offsets_2d = offsets.reshape(-1, block_size)
        self.block_starts = self.offsets_2d[:, 0]

    def locate(self, pos: int) -> int:
        """Return the index i such that offsets[i] <= pos < offsets[i+1]."""
        # Level 1: Find block
        block_idx = np.searchsorted(self.block_starts, pos, side="right") - 1
        block_idx = np.clip(block_idx, 0, self.offsets_2d.shape[0] - 1)

        # Level 2: Search within block
        row = self.offsets_2d[block_idx]
        within_idx = np.searchsorted(row, pos, side="right") - 1
        final_idx = block_idx * self.block_size + within_idx

        return min(final_idx, self.total_len - 1)

    def locate_batch(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized version of locate for multiple positions at once."""
        block_indices = np.searchsorted(self.block_starts, positions, side="right") - 1
        block_indices = np.clip(block_indices, 0, self.offsets_2d.shape[0] - 1)

        rows = self.offsets_2d[block_indices]
        within_indices = np.array(
            [np.searchsorted(row, pos, side="right") - 1 for row, pos in zip(rows, positions, strict=True)]
        )
        final_indices = block_indices * self.block_size + within_indices

        return np.minimum(final_indices, self.total_len - 1)

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens."""
        return int(self.offsets[-1])

    def get(self, idx: int | np.ndarray) -> int | np.ndarray:
        """Get the offset(s) at the given index or indices."""
        if np.isscalar(idx):
            if idx < 0 or idx >= len(self.offsets):
                raise IndexError(f"Index {idx} is out of bounds for offsets with length {len(self.offsets)}.")
            return int(self.offsets[idx])

        if np.any(idx < 0) or np.any(idx >= len(self.offsets)):
            raise IndexError(f"Some indices are out of bounds for offsets with length {len(self.offsets)}.")
        return self.offsets[idx]

    def __len__(self) -> int:
        return len(self.offsets)


class ResumableSampler(Sampler):
    """Sequential sampler that can resume from a saved position.

    Yields indices starting from start_index, wrapping around to 0
    when reaching the end of the dataset.
    """

    def __init__(self, data_source, start_index: int = 0):
        self.data_source = data_source
        self.start_index = start_index

    def __iter__(self):
        n = len(self.data_source)
        # Start from saved position, wrap around
        for i in range(n):
            yield (self.start_index + i) % n

    def __len__(self):
        return len(self.data_source)


class PackedTokenDataset(TorchDataset):
    """A map-style dataset that packs tokenized documents into fixed-length sequences.

    The packing works by:
    1. Shuffling documents (if shuffle_seed is provided)
    2. Computing cumulative offsets (document boundaries)
    3. Slicing the concatenated token stream into fixed-length sequences
    4. Optionally shuffling the resulting sequences

    Args:
        data_path: Path to a HuggingFace Arrow dataset with 'input_ids' column.
        seq_len: Sequence length for training (will be incremented by 1 for next-token prediction).
        eos_token_id: Token ID to insert between documents.
        shuffle_seed: Seed for shuffling (None = no shuffling).
    """

    _idx_dtype: np.typing.DTypeLike = np.uint32
    _offsets_dtype: np.typing.DTypeLike = np.uint64

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        eos_token_id: int = 0,
        shuffle_seed: int | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.seq_len = seq_len + 1  # Add one for next-token prediction
        self.eos_token_id = eos_token_id
        self.shuffle_seed = shuffle_seed
        self._setup()

    def _setup(self) -> None:
        """Initialize dataset, create/load metadata files."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")

        self.dataset: Dataset = load_from_disk(str(self.data_path)).with_format("numpy")
        if "input_ids" not in self.dataset.column_names:
            raise ValueError(f"Dataset must have 'input_ids' column. Found: {self.dataset.column_names}")

        # Create/load document index (for shuffling)
        doc_idx_path = self._metadata_path("docs")
        if not doc_idx_path.exists():
            self._save_shuffled_arange(len(self.dataset), doc_idx_path)
        self.doc_idx = np.memmap(doc_idx_path, dtype=self._idx_dtype, mode="r")

        # Create/load offsets
        offsets_path = self._metadata_path("offsets")
        if not offsets_path.exists():
            self._save_offsets(offsets_path)
        offsets = np.memmap(offsets_path, dtype=self._offsets_dtype, mode="r")
        self.offsets = OffsetLocator(offsets, block_size=2048)

        # Compute number of sequences
        self.total_tokens = self.offsets.total_tokens
        self.num_sequences = self.total_tokens // self.seq_len

        # Create/load sequence index (for sequence shuffling)
        seq_idx_path = self._metadata_path("seqs")
        if not seq_idx_path.exists():
            self._save_shuffled_arange(self.num_sequences, seq_idx_path)
        self.seq_idx = np.memmap(seq_idx_path, dtype=self._idx_dtype, mode="r")

    def _metadata_path(self, name: str) -> Path:
        """Generate path for metadata files."""
        suffix = f"seed{self.shuffle_seed}" if self.shuffle_seed is not None else "noshuffle"
        suffix += f"_eos{self.eos_token_id}_seq{self.seq_len}.npy"
        return self.data_path / f"{name}_{suffix}"

    def _save_shuffled_arange(self, size: int, path: Path) -> None:
        """Create and save a (optionally shuffled) arange array."""
        arr = np.arange(size, dtype=self._idx_dtype)
        if self.shuffle_seed is not None:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(arr)

        memmap = np.memmap(path, dtype=self._idx_dtype, mode="w+", shape=(size,))
        memmap[:] = arr
        memmap.flush()

    def _save_offsets(self, path: Path) -> None:
        """Compute and save document boundary offsets."""
        doc_lens = pc.list_value_length(self.dataset.data.table["input_ids"]).to_numpy()
        doc_lens = doc_lens + 1  # Add 1 for EOS token
        doc_lens = doc_lens[self.doc_idx]  # Reorder based on shuffle

        memmap = np.memmap(path, dtype=self._offsets_dtype, mode="w+", shape=(len(doc_lens) + 1,))
        memmap[0] = 0
        np.cumsum(doc_lens, out=memmap[1:])
        memmap.flush()

    def _get_sequence(self, start_pos: int, end_pos: int) -> np.ndarray:
        """Retrieve tokens for a sequence spanning [start_pos, end_pos)."""
        tokens = np.empty(end_pos - start_pos, dtype=np.int64)
        pos = start_pos
        i = 0

        while pos < end_pos:
            # Find document for current position
            shuffled_doc_idx = self.offsets.locate(pos)
            doc_idx = int(self.doc_idx[shuffled_doc_idx])
            input_ids = self.dataset[doc_idx]["input_ids"]

            # Position within document
            doc_start = int(pos - self.offsets.get(shuffled_doc_idx))
            tokens_to_copy = min(len(input_ids) - doc_start, end_pos - pos)

            # Copy tokens
            tokens[i : i + tokens_to_copy] = input_ids[doc_start : doc_start + tokens_to_copy]
            i += tokens_to_copy
            pos += tokens_to_copy

            # Add EOS if document ends and more tokens needed
            if doc_start + tokens_to_copy == len(input_ids) and pos < end_pos:
                tokens[i] = self.eos_token_id
                i += 1
                pos += 1

        return tokens

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of bounds for {self.num_sequences} sequences.")

        # Map to possibly shuffled sequence index
        index = int(self.seq_idx[idx])
        start_pos = index * self.seq_len
        end_pos = start_pos + self.seq_len

        tokens = self._get_sequence(start_pos, end_pos)
        return {"input_ids": torch.from_numpy(tokens)}

    def state_dict(self) -> dict:
        """Return state for Lightning checkpoint resumption."""
        return {"num_sequences": self.num_sequences, "seq_len": self.seq_len}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from Lightning checkpoint."""
        pass

    def get_sampler(self, start_index: int = 0) -> ResumableSampler:
        """Get a resumable sampler for this dataset."""
        return ResumableSampler(self, start_index=start_index)


class SimpleTokenDataset(TorchDataset):
    """Simple dataset that loads pre-packed sequences directly.

    Use this when data is already packed into fixed-length sequences
    (e.g., preprocessed datasets with 'input_ids' of uniform length).
    """

    def __init__(self, data_path: str | Path, shuffle_seed: int | None = None) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")

        self.dataset: Dataset = load_from_disk(str(self.data_path)).with_format("torch")
        if "input_ids" not in self.dataset.column_names:
            raise ValueError(f"Dataset must have 'input_ids' column. Found: {self.dataset.column_names}")

        if shuffle_seed is not None:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.dataset[idx]["input_ids"]}


@dataclass
class DataloaderKwargs:
    """Dataloader keyword arguments."""

    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DataModule(LightningDataModule):
    """Lightning DataModule for language model training with on-the-fly packing."""

    train_ds: PackedTokenDataset | SimpleTokenDataset | None = None
    val_ds: PackedTokenDataset | SimpleTokenDataset | None = None
    test_ds: PackedTokenDataset | SimpleTokenDataset | None = None

    def __init__(
        self,
        train_data_path: str | Path | None = None,
        val_data_path: str | Path | None = None,
        test_data_path: str | Path | None = None,
        seq_len: int = 2048,
        eos_token_id: int = 0,
        shuffle_seed: int | None = 42,
        batch_size: int = 32,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_packing: bool = True,
    ) -> None:
        super().__init__()
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.val_data_path = Path(val_data_path) if val_data_path else None
        self.test_data_path = Path(test_data_path) if test_data_path else None
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.shuffle_seed = shuffle_seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_packing = use_packing
        self.save_hyperparameters()

    def _create_dataset(
        self, path: Path | None, shuffle_seed: int | None = None
    ) -> PackedTokenDataset | SimpleTokenDataset | None:
        if path is None:
            return None

        if self.use_packing:
            return PackedTokenDataset(
                data_path=path,
                seq_len=self.seq_len,
                eos_token_id=self.eos_token_id,
                shuffle_seed=shuffle_seed,
            )
        else:
            return SimpleTokenDataset(data_path=path, shuffle_seed=shuffle_seed)

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = self._create_dataset(self.train_data_path, self.shuffle_seed)
            self.val_ds = self._create_dataset(self.val_data_path, shuffle_seed=None)

        if stage in ("validate", None) and self.val_ds is None:
            self.val_ds = self._create_dataset(self.val_data_path, shuffle_seed=None)

        if stage in ("test", None):
            self.test_ds = self._create_dataset(self.test_data_path, shuffle_seed=None)

    def _dataloader_kwargs(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": True,
            "shuffle": False,  # Shuffling handled by dataset
        }

    def set_resume_batch(self, batch_idx: int) -> None:
        """Set the batch index to resume from."""
        self._resume_batch_idx = batch_idx

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise ValueError("Train dataset not initialized. Call setup() first.")
        start = getattr(self, "_resume_batch_idx", 0)
        if start > 0:
            print(f"Resuming dataloader from batch {start}")
        sampler = ResumableSampler(self.train_ds, start_index=start)
        kwargs = self._dataloader_kwargs()
        kwargs.pop("shuffle", None)
        return DataLoader(self.train_ds, batch_size=self.batch_size, sampler=sampler, **kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise ValueError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(self.val_ds, batch_size=self.eval_batch_size, **self._dataloader_kwargs())

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise ValueError("Test dataset not initialized. Call setup() first.")
        return DataLoader(self.test_ds, batch_size=self.eval_batch_size, **self._dataloader_kwargs())
