"""BaseTrainer ABC — template method pattern for the training lifecycle."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adas_infra.core.schemas.manifest import RunManifest


class BaseTrainer(abc.ABC):
    """Abstract base for all training engines (single-device, DDP, FSDP).

    The template method `fit()` orchestrates lifecycle calls; subclasses
    override `_setup`, `_run_loop`, and `_teardown`.
    Shared state (device, config) is owned here; per-engine mutable state
    lives in the subclass.
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._hooks: list[Any] = []

    def register_hook(self, hook: Any) -> None:
        """Append a training hook (called after every step and at epoch boundaries)."""
        self._hooks.append(hook)

    def fit(
        self,
        model: Any,
        train_data: Any,
        val_data: Any | None = None,
    ) -> "RunManifest":
        """Full training lifecycle. Returns a RunManifest for reproducibility."""
        self._setup(model, train_data)
        manifest = self._run_loop(model, train_data, val_data)
        self._teardown(model, manifest)
        return manifest

    # ── Template methods (must be implemented by subclasses) ─────────────────

    @abc.abstractmethod
    def _setup(self, model: Any, train_data: Any) -> None:
        """Initialise device, distributed process group, optimizer, scheduler."""

    @abc.abstractmethod
    def _run_loop(
        self,
        model: Any,
        train_data: Any,
        val_data: Any | None,
    ) -> "RunManifest":
        """Execute the main training loop; return a completed RunManifest."""

    @abc.abstractmethod
    def _teardown(self, model: Any, manifest: "RunManifest") -> None:
        """Save checkpoint, persist run manifest, destroy process group."""
