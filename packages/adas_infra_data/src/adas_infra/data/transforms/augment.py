"""Augmentation transforms applied during training (not at validation/test time)."""

from __future__ import annotations

from typing import Any

import numpy as np


class RandomHorizontalFlip:
    """Randomly flip both modalities horizontally with probability *p*."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self._p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._rng.random() < self._p:
            sample["iris"] = np.flip(sample["iris"], axis=-1).copy()
            sample["fingerprint"] = np.flip(sample["fingerprint"], axis=-1).copy()
        return sample


class RandomGaussianNoise:
    """Add Gaussian noise to both modalities."""

    def __init__(self, std: float = 0.02, seed: int | None = None) -> None:
        self._std = std
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        iris = sample["iris"].astype(np.float32)
        fp = sample["fingerprint"].astype(np.float32)
        iris += self._rng.normal(0, self._std, iris.shape).astype(np.float32)
        fp += self._rng.normal(0, self._std, fp.shape).astype(np.float32)
        sample["iris"] = np.clip(iris, -3.0, 3.0)
        sample["fingerprint"] = np.clip(fp, -3.0, 3.0)
        return sample


class ComposeTransforms:
    """Chain a sequence of transforms, applying them in order."""

    def __init__(self, transforms: list[Any]) -> None:
        self._transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self._transforms:
            sample = t(sample)
        return sample
