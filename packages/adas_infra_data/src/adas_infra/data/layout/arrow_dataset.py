"""ArrowBiometricDataset — PyTorch-compatible dataset backed by a PyArrow table.

Decodes image bytes → float32 tensors on __getitem__ so the Arrow table
stays in its zero-copy columnar form until the moment PyTorch needs a sample.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from PIL import Image
from torch.utils.data import Dataset

from adas_infra.core.errors import MissingModalityError

logger = logging.getLogger(__name__)

_IRIS_SIZE = (64, 64)
_FP_SIZE = (96, 96)


class ArrowBiometricDataset(Dataset):  # type: ignore[type-arg]
    """In-memory PyTorch dataset wrapping a PyArrow table.

    Each sample is a dict::

        {
            "iris":        Tensor(1, 64, 64)   float32 in [0, 1]
            "fingerprint": Tensor(1, 96, 96)   float32 in [0, 1]
            "label":       Tensor()            int64
            "subject_id":  str
            "sample_id":   str
        }
    """

    def __init__(
        self,
        table: pa.Table,
        split: str = "train",
        iris_size: tuple[int, int] = _IRIS_SIZE,
        fp_size: tuple[int, int] = _FP_SIZE,
    ) -> None:
        mask = pc.equal(table.column("split"), split)  # type: ignore[attr-defined]
        self._table = table.filter(mask)
        self._split = split
        self._iris_size = iris_size
        self._fp_size = fp_size
        logger.info("ArrowBiometricDataset: split=%s, rows=%d", split, len(self._table))

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Direct column access avoids the table.slice() allocation per sample.
        subject_id: str = self._table.column("subject_id")[idx].as_py()
        sample_id: str = self._table.column("sample_id")[idx].as_py()
        iris_raw: bytes | None = self._table.column("iris_bytes")[idx].as_py()
        fp_raw: bytes | None = self._table.column("fingerprint_bytes")[idx].as_py()
        label_val: int = self._table.column("label")[idx].as_py()

        if not iris_raw:
            raise MissingModalityError(subject_id, "iris", sample_id)
        if not fp_raw:
            raise MissingModalityError(subject_id, "fingerprint", sample_id)

        iris_tensor = self._decode_image(iris_raw, self._iris_size)
        fp_tensor = self._decode_image(fp_raw, self._fp_size)

        return {
            "iris": iris_tensor,
            "fingerprint": fp_tensor,
            "label": torch.tensor(label_val, dtype=torch.int64),
            "subject_id": subject_id,
            "sample_id": sample_id,
        }

    @staticmethod
    def _decode_image(raw: bytes, size: tuple[int, int]) -> torch.Tensor:
        """Decode raw image bytes → resized grayscale float32 tensor [0, 1]."""
        img = Image.open(io.BytesIO(raw)).convert("L").resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    def num_classes(self) -> int:
        """Return the number of unique subject labels in this split."""
        return int(pc.max(self._table.column("label")).as_py()) + 1  # type: ignore[attr-defined]
