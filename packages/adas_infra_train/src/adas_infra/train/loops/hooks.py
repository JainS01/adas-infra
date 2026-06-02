"""Training lifecycle hooks.

Hooks implement a simple event-based interface:
  on_step_end(step, metrics)      — called after every optimiser step
  on_epoch_end(epoch, metrics)    — called after every epoch
  on_train_end(manifest)          — called once; receives final RunManifest

RunManifestHook is the most critical hook: it persists the reproducibility
4-tuple to MLflow tags AND to run_manifest.json (§0.7). If either write fails,
the hook raises ReproducibilityError which aborts the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from adas_infra.core.errors import ReproducibilityError
from adas_infra.core.schemas.manifest import RunManifest

logger = logging.getLogger(__name__)


class TrainingHook:
    """No-op base class; subclass and override the events you care about."""

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_train_end(self, manifest: RunManifest) -> None:
        pass


class LoggingHook(TrainingHook):
    """Emits step and epoch metrics to the Python logger."""

    def __init__(self, log_every_n_steps: int = 10) -> None:
        self._log_every = log_every_n_steps

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        if step % self._log_every == 0:
            metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info("step %d — %s", step, metric_str)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("epoch %d — %s", epoch, metric_str)


class RunManifestHook(TrainingHook):
    """Persists the run manifest as MLflow tags AND run_manifest.json.

    Both writes happen atomically inside on_train_end; failure of either
    raises ReproducibilityError which aborts the run (§0.7).
    """

    def __init__(self, checkpoint_dir: Path, mlflow_run_id: str | None = None) -> None:
        self._ckpt_dir = checkpoint_dir
        self._mlflow_run_id = mlflow_run_id

    def on_train_end(self, manifest: RunManifest) -> None:
        self._write_json(manifest)
        self._write_mlflow_tags(manifest)

    def _write_json(self, manifest: RunManifest) -> None:
        dest = self._ckpt_dir / "run_manifest.json"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(manifest.model_dump_json(indent=2))
            logger.info("RunManifestHook: wrote %s", dest)
        except OSError as exc:
            raise ReproducibilityError(f"Cannot write run_manifest.json: {exc}") from exc

    def _write_mlflow_tags(self, manifest: RunManifest) -> None:
        try:
            import mlflow

            run_id = self._mlflow_run_id or manifest.run_id
            client = mlflow.tracking.MlflowClient()
            tags = manifest.as_mlflow_tags()
            for key, value in tags.items():
                client.set_tag(run_id, key, value)
            logger.info("RunManifestHook: set %d MLflow tags on run %s", len(tags), run_id)
        except (ImportError, OSError, RuntimeError) as exc:
            raise ReproducibilityError(f"Cannot write MLflow tags: {exc}") from exc


class CheckpointHook(TrainingHook):
    """Saves a checkpoint every *save_every_n_steps* steps and at run end."""

    def __init__(
        self,
        checkpoint_dir: Path,
        model: Any,
        optimizer: Any,
        save_every_n_steps: int = 100,
    ) -> None:
        self._ckpt_dir = checkpoint_dir
        self._model = model
        self._optimizer = optimizer
        self._save_every = save_every_n_steps

    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        if step > 0 and step % self._save_every == 0:
            self._save(step, metrics)

    def on_train_end(self, manifest: RunManifest) -> None:
        self._save(step=-1, metrics={})

    def _save(self, step: int, metrics: dict[str, float]) -> None:
        import torch

        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = f"step_{step:06d}" if step >= 0 else "final"
        path = self._ckpt_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("CheckpointHook: saved %s", path)
