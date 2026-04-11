"""
Data loading for pretraining.

The tokenisation pipeline produced flat binary files (train.bin, val.bin)
of concatenated token-ids with <|endoftext|> delimiters.  We slice them
into fixed-length windows of `block_size` tokens for causal-LM training.

Because the whole dataset fits in RAM we memory-map the file once and let
every worker read from the same shared ndarray (copy-on-write via mmap),
which avoids duplicating 5+ GB per DataLoader worker.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path


class PretrainDataset(Dataset):
    """
    Slices a flat token array into (input, target) pairs of length `block_size`.

    token array:  [t0, t1, t2, ..., tN]
    sample i:
        x = tokens[i * block_size : (i+1) * block_size]
        y = tokens[i * block_size + 1 : (i+1) * block_size + 1]

    Consecutive windows with stride = block_size (no overlap) is standard for
    pretraining; overlap would waste compute on repeated tokens.
    """

    def __init__(self, bin_path: str | Path, block_size: int, dtype=np.uint16):
        self.block_size = block_size
        # Memory-map: the OS pages data in on demand; forked workers share pages.
        self.data = np.memmap(str(bin_path), dtype=dtype, mode="r")
        # Number of complete, non-overlapping windows we can form.
        # We need block_size + 1 tokens per sample (input + 1 shifted target).
        self.n_samples = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size
        # .copy() so the returned tensor owns its memory (avoids mmap issues)
        x = torch.from_numpy(self.data[start:end].astype(np.int64).copy())
        y = torch.from_numpy(self.data[start + 1:end + 1].astype(np.int64).copy())
        return x, y


def create_dataloaders(
    data_dir: str | Path,
    block_size: int,
    micro_batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    dtype=np.uint16,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    With DDP each rank sees a disjoint shard of the data via DistributedSampler.
    """
    data_dir = Path(data_dir)
    train_ds = PretrainDataset(data_dir / "train.bin", block_size, dtype=dtype)
    val_ds = PretrainDataset(data_dir / "val.bin", block_size, dtype=dtype)

    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, seed=seed, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        shuffle_train = False   # sampler handles shuffling

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,         # avoid ragged last batch
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

