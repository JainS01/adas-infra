"""LocalCkptStore — save and load model checkpoints from the local filesystem."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from adas_infra.core.errors import CheckpointError

logger = logging.getLogger(__name__)


class LocalCkptStore:
    """Manages checkpoint save/load for single-device and DDP training.

    For FSDP, use `dist_cp_adapter.py` (sharded checkpoints); this store
    handles the simpler case of a single state_dict.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        step: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self._dir / f"ckpt_step{step:06d}.pt"
        try:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    **(extra or {}),
                },
                path,
            )
        except OSError as exc:
            raise CheckpointError(f"Failed to save checkpoint to {path}: {exc}") from exc
        logger.info("LocalCkptStore: saved step %d → %s", step, path)
        return path

    def load_latest(self, model: nn.Module, optimizer: Any | None = None) -> dict[str, Any]:
        """Load the most recently saved checkpoint by step number."""
        checkpoints = sorted(self._dir.glob("ckpt_step*.pt"))
        if not checkpoints:
            raise CheckpointError(f"No checkpoints found in {self._dir}")
        return self.load(checkpoints[-1], model, optimizer)

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Any | None = None,
    ) -> dict[str, Any]:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointError(f"Failed to load checkpoint {path}: {exc}") from exc

        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info("LocalCkptStore: loaded step %d from %s", state.get("step", -1), path)
        return state  # type: ignore[no-any-return]

    def export_for_registry(self, model: nn.Module, dest: Path) -> Path:
        """Save the model state_dict-only to *dest* for registry upload."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, dest)
        logger.info("LocalCkptStore: exported weights to %s", dest)
        return dest
