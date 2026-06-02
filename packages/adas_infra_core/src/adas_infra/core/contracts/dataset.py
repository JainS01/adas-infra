"""MultimodalDataset protocol — PyTorch-compatible dataset abstraction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MultimodalDataset(Protocol):
    """A protocol compatible with torch.utils.data.Dataset for multimodal samples.

    Each item is a dict with at least:
      iris         : torch.Tensor  shape (1, H, W)  float32 in [0, 1]
      fingerprint  : torch.Tensor  shape (1, H, W)  float32 in [0, 1]
      label        : torch.Tensor  shape ()          int64
      subject_id   : str
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...
