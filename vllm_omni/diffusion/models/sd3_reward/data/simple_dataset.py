from __future__ import annotations

import glob
import io
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


class SimpleParquetDataset(Dataset):
    def __init__(
        self,
        parquet_files: Sequence[str],
        process_fn: Callable[[Any], dict[str, Any]] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        self.parquet_files = list(parquet_files)
        self.process_fn = process_fn or self._default_process_fn
        frames = [pd.read_parquet(parquet_file) for parquet_file in self.parquet_files]
        if not frames:
            raise ValueError("SimpleParquetDataset requires at least one parquet file")

        self.data = pd.concat(frames, ignore_index=True)
        self.total_samples = len(self.data)

        if shuffle:
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)

    def _default_process_fn(self, row: Any) -> dict[str, Any]:
        return {
            "prompt": row["prompt"],
            "latent_chosen": torch.from_numpy(npy_bytes_to_ndarray(row["latent1"])),
            "latent_reject": torch.from_numpy(npy_bytes_to_ndarray(row["latent2"])),
        }

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index: int):
        return self.process_fn(self.data.iloc[index])


def create_simple_dataloader(
    parquet_files: str | Sequence[str],
    process_fn: Callable[[Any], dict[str, Any]] | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    **kwargs,
) -> DataLoader:
    if isinstance(parquet_files, str):
        parquet_files = sorted(glob.glob(parquet_files))

    dataset = SimpleParquetDataset(
        parquet_files=parquet_files,
        process_fn=process_fn,
        shuffle=shuffle,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
