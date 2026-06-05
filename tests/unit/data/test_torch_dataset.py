"""Unit tests for the standard-DataLoader fallback path.

Exercises ArrowBiometricDataset (the concrete MultimodalDataset) and
build_torch_dataloader (typed against the Protocol contract + ShardedSampler).
"""

from __future__ import annotations

import torch

from adas_infra.core.contracts.dataset import MultimodalDataset
from adas_infra.data.loaders.torch_dataset import (
    ArrowBiometricDataset,
    build_torch_dataloader,
)


class TestArrowBiometricDataset:
    def test_satisfies_multimodal_dataset_protocol(self, synthetic_table):
        ds = ArrowBiometricDataset(synthetic_table)
        # runtime_checkable Protocol — structural conformance, the whole point.
        assert isinstance(ds, MultimodalDataset)

    def test_len_matches_table(self, synthetic_table):
        ds = ArrowBiometricDataset(synthetic_table)
        assert len(ds) == synthetic_table.num_rows

    def test_getitem_shapes_and_dtypes(self, synthetic_table):
        ds = ArrowBiometricDataset(synthetic_table)
        item = ds[0]
        assert set(item) >= {"iris", "fingerprint", "label", "subject_id"}
        assert item["iris"].shape == (1, 64, 64)
        assert item["fingerprint"].shape == (1, 96, 96)
        assert item["label"].dtype == torch.int64

    def test_build_dataloader_yields_full_batches(self, synthetic_table):
        ds = ArrowBiometricDataset(synthetic_table)
        dl = build_torch_dataloader(ds, batch_size=4, shuffle=False, drop_last=True)
        batch = next(iter(dl))
        assert batch["iris"].shape == (4, 1, 64, 64)
        assert batch["fingerprint"].shape == (4, 1, 96, 96)
        assert batch["label"].shape == (4,)
