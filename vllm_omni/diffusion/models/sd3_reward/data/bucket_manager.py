# Modified from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import tqdm

CUSTOM_BUCKETS = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1248, 832),
    (1344, 896),
    (832, 1248),
    (896, 1344),
    (1024, 768),
    (768, 1024),
    (1472, 832),
    (832, 1472),
    (1152, 832),
    (832, 1152),
]


def get_prng(seed):
    return np.random.RandomState(seed)


class BucketManager:
    def __init__(
        self,
        bucket_file=None,
        divisible=32,
        ar_thresh=0.03,
        base_resolution=(1024, 1024),
        bsz=3,
        world_size=1,
        global_rank=0,
        seed=42,
        use_dynamic_bsz=True,
        debug=False,
        num_workers: int = 1,
        worker_id: int = 0,
    ):
        self.div = divisible
        self.ar_thresh = ar_thresh
        self.debug = debug

        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank

        self.num_workers = num_workers
        self.worker_id = worker_id

        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32 - 1)
        self.epoch_prng = get_prng(epoch_seed)

        self.resolutions = None
        self.aspects = None
        self.areas = None

        self.res_map: dict[int, tuple[int, int]] = {}
        self.sample_info: dict[int, dict[str, Any]] = {}
        self.buckets: dict[int, list[int]] = {}

        self.epoch = None
        self.batch_total = 0
        self.batch_delivered = 0
        self._batch_queue = []

        self.gen_buckets()
        if use_dynamic_bsz:
            base_area = base_resolution[0] * base_resolution[1]
            bucket_areas = self.areas
            self.bucket_bsz = np.maximum(1, (base_area // bucket_areas) * self.bsz)
        else:
            self.bucket_bsz = np.full((len(self.resolutions),), self.bsz, dtype=np.int32)

        if bucket_file is not None:
            self.load_from_csv(bucket_file)
            if len(self.res_map) > 0:
                self.build_buckets()
                self.start_epoch()

    def set_worker_info(self, num_workers: int, worker_id: int):
        self.num_workers = num_workers
        self.worker_id = worker_id

    def load_from_csv(self, bucket_file):
        res_to_id = {(int(h), int(w)): i for i, (h, w) in enumerate(self.resolutions)}
        df = pd.read_csv(bucket_file)
        df["bucket_id"] = df.apply(lambda row: res_to_id.get((int(row["height"]), int(row["width"]))), axis=1)
        missed = df["bucket_id"].isnull().sum()
        if missed > 0:
            print(f"Warning: {missed} entries in bucket index could not be matched to any bucket.")
        df = df.dropna(subset=["bucket_id"])

        for row in df.itertuples(index=False):
            gid = int(row.global_id)
            width = int(row.width)
            height = int(row.height)
            bucket_id = int(row.bucket_id)

            self.res_map[gid] = (height, width)
            self.sample_info[gid] = {
                "global_id": gid,
                "parquet_path": row.parquet_path,
                "row_group": int(row.row_group),
                "row_index": int(row.row_index),
                "bucket_id": bucket_id,
                "width": width,
                "height": height,
            }

        if self.debug:
            print(f"[load_from_csv] loaded {len(self.res_map)} items")

    def gen_buckets(self):
        normed = []
        seen = set()
        for height, width in CUSTOM_BUCKETS:
            height, width = int(height), int(width)
            if height <= 0 or width <= 0:
                continue
            if (height, width) in seen:
                continue
            if self.div is not None and (height % self.div != 0 or width % self.div != 0):
                print(f"Warning: bucket {(height, width)} is not divisible by {self.div}, rounding down.")
                height = (height // self.div) * self.div
                width = (width // self.div) * self.div
            seen.add((height, width))
            normed.append((height, width))

        if not normed:
            raise ValueError("No valid buckets remain after filtering. Check CUSTOM_BUCKETS / constraints.")

        ordered = sorted(normed, key=lambda x: x[0] * x[1])
        self.resolutions = np.array(ordered, dtype=np.int32)
        self.aspects = np.array([height / float(width) for (height, width) in ordered], dtype=np.float32)
        self.areas = np.array([height * width for (height, width) in ordered], dtype=np.int32)

        if self.debug:
            print(f"resolutions:\n{self.resolutions}")

    def assign_bucket(self, res, ar_thresh=None, return_res=False):
        height, width = int(res[0]), int(res[1])
        aspect = height / float(width)
        ar_errors = np.abs(np.log(self.aspects) - np.log(aspect))

        threshold = ar_thresh or self.ar_thresh
        candidates = np.where(ar_errors <= threshold)[0]
        if len(candidates) == 0:
            return None

        area = height * width
        candidate_areas = self.areas[candidates]
        area_errors = np.abs(np.log(candidate_areas) - np.log(area))
        best_candidate = candidates[np.argmin(area_errors)]

        if return_res:
            return tuple(self.resolutions[best_candidate])
        return best_candidate

    def build_buckets(self):
        self.buckets = {}
        skipped = 0
        self.aspect_errors = []

        for post_id, (height, width) in self.res_map.items():
            bucket_id = self.assign_bucket((height, width))
            if bucket_id is None:
                skipped += 1
                continue
            self.buckets.setdefault(bucket_id, []).append(post_id)
            if self.debug:
                aspect = float(height) / float(width)
                self.aspect_errors.append(abs(self.aspects[bucket_id] - aspect))

        if self.debug:
            self.aspect_errors = np.array(self.aspect_errors) if self.aspect_errors else np.array([0.0])
            print(f"[build_buckets] skipped: {skipped}")
            for bucket_id in sorted(self.buckets.keys()):
                print(f"  bucket {bucket_id} {tuple(self.resolutions[bucket_id])}: {len(self.buckets[bucket_id])} items")
            print(
                f"[build_buckets] aspect error mean={self.aspect_errors.mean():.4f}, "
                f"median={np.median(self.aspect_errors):.4f}, max={self.aspect_errors.max():.4f}"
            )
            print(f"[build_buckets] total buckets used: {len(self.buckets)}")
            print(f"[build_buckets] total items assigned: {sum(len(v) for v in self.buckets.values())}")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            t0 = time.perf_counter()

        if world_size is not None:
            self.world_size = int(world_size)
        if global_rank is not None:
            self.global_rank = int(global_rank)

        batch_entries = []
        for bucket_id in tqdm.tqdm(sorted(self.buckets.keys()), desc="Preparing buckets"):
            bucket_bs = self.bucket_bsz[bucket_id]
            global_bucket_bs = bucket_bs * self.world_size

            arr = np.array(self.buckets[bucket_id], dtype=np.int64)
            if arr.size == 0:
                continue

            self.prng.shuffle(arr)

            usable_len = (arr.size // global_bucket_bs) * global_bucket_bs
            if usable_len == 0:
                continue
            arr = arr[:usable_len]

            arr_rank = arr[self.global_rank::self.world_size]
            n_full = (arr_rank.size // bucket_bs) * bucket_bs
            arr_rank = arr_rank[:n_full]

            for i in range(0, n_full, bucket_bs):
                batch_ids = arr_rank[i : i + bucket_bs].tolist()
                batch_entries.append((bucket_id, bucket_bs, tuple(self.resolutions[bucket_id]), batch_ids))

        if not batch_entries:
            raise RuntimeError("No valid batches produced; try reducing batch sizes or merging buckets.")

        perm = self.epoch_prng.permutation(len(batch_entries))
        self._batch_queue = [batch_entries[i] for i in perm]

        self.batch_total = len(self._batch_queue)
        self.batch_delivered = 0
        self.epoch = True

        if self.debug:
            print(f"[start_epoch] rank={self.global_rank}: {self.batch_total} batches, time={time.perf_counter() - t0:.4f}s")

    def generator(self):
        if not self._batch_queue or self.batch_delivered is None or self.batch_delivered >= self.batch_total:
            self.start_epoch()
        for batch_idx in range(self.batch_total):
            if batch_idx % self.num_workers != self.worker_id:
                continue

            _, _, resolution, batch_ids = self._batch_queue[batch_idx]
            batch_sample_infos = [self.sample_info[pid] for pid in batch_ids]

            yield (batch_ids, resolution, batch_sample_infos)
