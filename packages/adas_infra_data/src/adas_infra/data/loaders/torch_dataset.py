"""ArrowBiometricDataset — map-style torch Dataset over an in-memory Arrow table.

This is the standard-DataLoader fallback data path used by the single-device /
non-Plasma profile (the Ray/Plasma path is the high-throughput default). It is the
concrete realisation of the ``MultimodalDataset`` contract: the training stack and
``build_torch_dataloader`` bind to the Protocol, never to this class, so a future
on-disk or memory-mapped dataset can drop in without touching callers.
"""

from __future__ import annotations

import io
from typing import Any, cast

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from adas_infra.core.contracts.dataset import MultimodalDataset
from adas_infra.data.loaders.sharded_sampler import ShardedSampler

_IRIS_SIZE = (64, 64)
_FP_SIZE = (96, 96)


def _decode(raw: bytes, size: tuple[int, int]) -> torch.Tensor:
    """Decode image bytes to a (1, H, W) float32 tensor in [0, 1]."""
    img = Image.open(io.BytesIO(raw)).convert("L").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class ArrowBiometricDataset(Dataset[dict[str, Any]]):
    """Wraps a BiometricFrame Arrow table as an indexable torch Dataset.

    Each item matches the ``MultimodalDataset`` contract: a dict with ``iris`` and
    ``fingerprint`` tensors of shape (1, H, W), an int64 ``label``, and ``subject_id``.
    """

    def __init__(self, table: pa.Table) -> None:
        self._iris: list[bytes] = table.column("iris_bytes").to_pylist()
        self._fp: list[bytes] = table.column("fingerprint_bytes").to_pylist()
        self._label: list[int] = table.column("label").to_pylist()
        self._subject: list[str] = table.column("subject_id").to_pylist()

    def __len__(self) -> int:
        return len(self._label)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "iris": _decode(self._iris[idx], _IRIS_SIZE),
            "fingerprint": _decode(self._fp[idx], _FP_SIZE),
            "label": torch.tensor(self._label[idx], dtype=torch.int64),
            "subject_id": str(self._subject[idx]),
        }


def build_torch_dataloader(
    dataset: MultimodalDataset,
    *,
    batch_size: int = 32,
    rank: int | None = None,
    world_size: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> DataLoader[dict[str, Any]]:
    """Build a DDP-aware DataLoader over any ``MultimodalDataset``.

    Pairs the dataset contract with :class:`ShardedSampler` so each rank draws a
    non-overlapping, deterministically-shuffled subset. ``drop_last=True`` keeps
    every batch full — important because the fusion head's BatchNorm1d rejects a
    batch of size 1. The ``dataset`` parameter is typed to the Protocol, not the
    concrete class, which is the whole point of the abstraction.
    """
    sampler = ShardedSampler(
        dataset_size=len(dataset),
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
    # DataLoader wants a torch Dataset; MultimodalDataset is structurally compatible
    # (it has __len__/__getitem__). The cast documents that equivalence for mypy.
    return DataLoader(
        cast("Dataset[dict[str, Any]]", dataset),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
    )
