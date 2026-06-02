"""Evaluation loop — forward-only pass over the validation set."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EvalLoop:
    """Runs a full forward pass over *data* and computes aggregate metrics.

    Returns a dict: {"val_loss": float, "val_accuracy": float}
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._criterion = nn.CrossEntropyLoss()

    def run(self, data: Iterable[dict[str, Any]]) -> dict[str, float]:
        self._model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data:
                iris = self._to_device(batch["iris"])
                fingerprint = self._to_device(batch["fingerprint"])
                labels = self._to_device(batch["label"]).long()

                logits = self._model(iris, fingerprint)
                loss = self._criterion(logits, labels)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_correct += (logits.argmax(-1) == labels).sum().item()
                total_samples += bs

        if total_samples == 0:
            logger.warning("EvalLoop: no samples evaluated")
            return {"val_loss": float("nan"), "val_accuracy": float("nan")}

        return {
            "val_loss": total_loss / total_samples,
            "val_accuracy": total_correct / total_samples,
        }

    def _to_device(self, tensor: Any) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self._device, non_blocking=True)
        return torch.tensor(tensor, device=self._device)
