from __future__ import annotations

import glob
import io
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from vllm_omni.diffusion.models.sd3_reward.data.bucket_manager import BucketManager


def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


class BucketDataset(IterableDataset):
    def __init__(
        self,
        parquet_files: Sequence[str],
        bucket_manager: BucketManager,
    ):
        super().__init__()
        self.bucket_manager = bucket_manager
        self.total_samples = len(self.bucket_manager.res_map)
        self.parquet_files = {os.path.basename(f): f for f in parquet_files}

    def _process_sample(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": row["prompt"],
            "latent_chosen": torch.from_numpy(npy_bytes_to_ndarray(row["latent1"])),
            "latent_reject": torch.from_numpy(npy_bytes_to_ndarray(row["latent2"])),
        }

    def __len__(self):
        return self.bucket_manager.batch_total * self.bucket_manager.world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        self.bucket_manager.set_worker_info(
            worker_id=worker_id,
            num_workers=num_workers,
        )

        for _, _, sample_infos in self.bucket_manager.generator():
            batch_data = []
            for sample_info in sample_infos:
                try:
                    parquet_file = os.path.basename(sample_info["parquet_path"])
                    row_group = sample_info["row_group"]
                    row_id = sample_info["row_index"]
                    pf = pq.ParquetFile(self.parquet_files[parquet_file])

                    row = pf.read_row_group(row_group, columns=["prompt", "latent1", "latent2"])
                    data = self._process_sample({
                        "prompt": row.column("prompt")[row_id].as_py(),
                        "latent1": row.column("latent1")[row_id].as_py(),
                        "latent2": row.column("latent2")[row_id].as_py(),
                    })
                    batch_data.append(data)
                except Exception as e:
                    print(f"Error processing sample {sample_info['global_id']}: {e}")
            yield batch_data


def collate_fn(batch_list: list[list[dict[str, Any]]]) -> dict[str, Any]:
    if len(batch_list) != 1:
        raise ValueError(f"Expected batch_list length 1, got {len(batch_list)}")

    batch = batch_list[0]

    if not batch:
        return {}
    collated = {
        "prompt": [item["prompt"] for item in batch],
        "latent_chosen": torch.stack([item["latent_chosen"] for item in batch]),
        "latent_reject": torch.stack([item["latent_reject"] for item in batch]),
    }

    return collated


class BucketDataLoader:
    def __init__(
        self,
        parquet_files: Sequence[str],
        bucket_file: str,
        base_resolution=(1024, 1024),
        bsz=3,
        world_size=1,
        global_rank=0,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        use_dynamic_bsz: bool = True,
    ):
        self.bucket_manager = BucketManager(
            bucket_file=bucket_file,
            divisible=32,
            ar_thresh=0.03,
            base_resolution=base_resolution,
            bsz=bsz,
            world_size=world_size,
            global_rank=global_rank,
            seed=seed,
            use_dynamic_bsz=use_dynamic_bsz,
            debug=False,
        )

        self.dataset = BucketDataset(
            parquet_files=parquet_files,
            bucket_manager=self.bucket_manager,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        self.epoch_count = 0

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataset)

    def start_epoch(self):
        self.epoch_count += 1
        self.bucket_manager.start_epoch()


def create_bucket_dataloader(
    parquet_files: str | Sequence[str],
    bucket_file: str,
    reference_size=1024,
    base_batch_size=3,
    world_size=1,
    global_rank=0,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    use_dynamic_bsz: bool = True,
) -> BucketDataLoader:
    if isinstance(parquet_files, str):
        parquet_files = sorted(glob.glob(parquet_files))

    return BucketDataLoader(
        parquet_files=parquet_files,
        bucket_file=bucket_file,
        base_resolution=(reference_size, reference_size),
        bsz=base_batch_size,
        world_size=world_size,
        global_rank=global_rank,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        use_dynamic_bsz=use_dynamic_bsz,
    )
