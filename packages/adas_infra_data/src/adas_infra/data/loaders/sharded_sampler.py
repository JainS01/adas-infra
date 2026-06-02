"""ShardedSampler — DDP-aware distributed sampler for PyTorch DataLoader.

Partitions shard indices across ranks so each GPU processes a non-overlapping
subset of shards. Complements PlasmaPrefetcher: use ShardedSampler when
falling back to a standard DataLoader (e.g., in the single-device profile).
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class ShardedSampler(Sampler[int]):
    """Distributes indices evenly across distributed training ranks.

    Pads the dataset so every rank gets exactly `ceil(N/world_size)` samples.
    Drop_last=True removes the padding on the last rank for precise epoch counting.
    """

    def __init__(
        self,
        dataset_size: int,
        rank: int | None = None,
        world_size: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self._dataset_size = dataset_size
        self._rank = rank
        self._world_size = world_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0

        if drop_last and dataset_size % world_size != 0:
            self._num_samples = math.floor(dataset_size / world_size)
        else:
            self._num_samples = math.ceil(dataset_size / world_size)

        self._total_size = self._num_samples * world_size

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for deterministic per-epoch shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)

        if self._shuffle:
            indices = torch.randperm(self._dataset_size, generator=g).tolist()
        else:
            indices = list(range(self._dataset_size))

        if not self._drop_last:
            padding_size = self._total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self._total_size]

        rank_indices = indices[self._rank : self._total_size : self._world_size]
        assert len(rank_indices) == self._num_samples
        return iter(rank_indices)

    def __len__(self) -> int:
        return self._num_samples
