"""MultimodalAlignTransform — normalises and resizes both modalities to fixed shapes.

Applied in the Ray actor process (inside _decode_and_align_batch) and also
available as a standalone callable for the fallback DataLoader path.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class MultimodalAlignConfig:
    iris_h: int = 64
    iris_w: int = 64
    fp_h: int = 96
    fp_w: int = 96
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


class MultimodalAlignTransform:
    """Stateless transform that decodes image bytes and aligns spatial dimensions.

    Can be used in two modes:
      1. As a Ray map_batches callable (receives dict of numpy arrays / bytes lists)
      2. As a per-sample callable in ArrowBiometricDataset.__getitem__
    """

    def __init__(self, cfg: MultimodalAlignConfig | None = None) -> None:
        self._cfg = cfg or MultimodalAlignConfig()

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Process a batch dict with iris_bytes and fingerprint_bytes columns."""
        iris_list = []
        fp_list = []

        for iris_raw, fp_raw in zip(batch["iris_bytes"], batch["fingerprint_bytes"], strict=False):
            iris_list.append(self._decode(bytes(iris_raw), self._cfg.iris_h, self._cfg.iris_w))
            fp_list.append(self._decode(bytes(fp_raw), self._cfg.fp_h, self._cfg.fp_w))

        result = dict(batch)
        result["iris"] = np.stack(iris_list, axis=0)  # (B, 1, H, W)
        result["fingerprint"] = np.stack(fp_list, axis=0)  # (B, 1, H, W)
        return result

    def _decode(self, raw: bytes, h: int, w: int) -> np.ndarray:
        img = Image.open(io.BytesIO(raw)).convert("L").resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        # z-score normalise around 0.5
        arr = (arr - self._cfg.normalize_mean) / self._cfg.normalize_std
        return arr[np.newaxis, :, :]  # (1, H, W)
