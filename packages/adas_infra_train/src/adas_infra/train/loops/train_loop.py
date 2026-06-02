"""Main training loop — step-level forward/backward/optimise cycle.

The loop is agnostic to the data source: it accepts any iterable of batch
dicts (PlasmaPrefetcher, DataLoader, or a plain list for unit tests).
Hook callbacks fire at every step and at epoch boundaries.

Metrics tracked per step:
  loss        — cross-entropy on the current batch
  accuracy    — top-1 accuracy on the current batch
  lr          — current learning rate (from the last param group)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from adas_infra.train.loops.hooks import TrainingHook

logger = logging.getLogger(__name__)

# Bottleneck diagnosis: if data_load_fraction exceeds this threshold the
# preprocessing pipeline (Plasma prefetcher) is the bottleneck, not the GPU.
_DATA_LOAD_BOTTLENECK_THRESHOLD = 0.3  # 30 % of wall time spent waiting for batches


class TrainLoop:
    """Runs the forward-backward cycle for *max_steps* steps.

    Args:
        model:       PyTorch module (already on *device*)
        optimizer:   PyTorch optimizer
        scheduler:   Optional LR scheduler (step() called after each batch)
        device:      torch.device
        max_steps:   Hard stop; -1 means run until data exhausted
        hooks:       List of TrainingHook instances
        amp:         Whether to use torch.amp.autocast (disabled on CPU)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any | None,
        device: torch.device,
        max_steps: int = -1,
        hooks: list[TrainingHook] | None = None,
        amp: bool = False,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._max_steps = max_steps
        self._hooks = hooks or []
        self._amp = amp and device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda") if self._amp else None
        self._criterion = nn.CrossEntropyLoss()

    def run(self, data: Iterable[dict[str, Any]]) -> dict[str, float]:
        """Iterate over *data* for up to *max_steps* steps.

        Returns the final-step metrics dict, augmented with timing breakdown:
          data_load_time_s   — cumulative seconds waiting for the next batch
          compute_time_s     — cumulative seconds in forward + backward
          data_load_fraction — data_load / total_wall_time (bottleneck signal)
        """
        self._model.train()
        step = 0
        last_metrics: dict[str, float] = {}
        total_data_load_s = 0.0
        total_compute_s = 0.0
        t_wall_start = time.perf_counter()

        data_iter = iter(data)
        while True:
            if self._max_steps >= 0 and step >= self._max_steps:
                break

            # ── Time the data-loading wait separately ───────────────────────
            t_load_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            total_data_load_s += time.perf_counter() - t_load_start

            # ── Time the forward + backward compute ─────────────────────────
            t_compute_start = time.perf_counter()
            metrics = self._step(batch, step)
            total_compute_s += time.perf_counter() - t_compute_start

            last_metrics = metrics
            for hook in self._hooks:
                hook.on_step_end(step, metrics)
            step += 1

        total_wall_s = time.perf_counter() - t_wall_start
        data_load_fraction = total_data_load_s / max(total_wall_s, 1e-6)

        logger.info(
            "TrainLoop: %d steps | wall=%.2fs | data_load=%.2fs (%.0f%%) | "
            "compute=%.2fs (%.0f%%) | throughput=%.1f steps/s",
            step,
            total_wall_s,
            total_data_load_s,
            data_load_fraction * 100,
            total_compute_s,
            total_compute_s / max(total_wall_s, 1e-6) * 100,
            step / max(total_wall_s, 1e-6),
        )

        if data_load_fraction > _DATA_LOAD_BOTTLENECK_THRESHOLD:
            logger.warning(
                "BOTTLENECK DETECTED: %.0f%% of wall time is data loading. "
                "Consider increasing PlasmaPrefetcher queue depth or Ray concurrency.",
                data_load_fraction * 100,
            )

        last_metrics.update(
            {
                "data_load_time_s": total_data_load_s,
                "compute_time_s": total_compute_s,
                "data_load_fraction": data_load_fraction,
            }
        )
        return last_metrics

    def _step(self, batch: dict[str, Any], step: int) -> dict[str, float]:
        iris = self._to_device(batch["iris"])        # (B, 1, H_iris, W_iris)
        fingerprint = self._to_device(batch["fingerprint"])  # (B, 1, H_fp, W_fp)
        labels = self._to_device(batch["label"]).long()

        self._optimizer.zero_grad(set_to_none=True)

        if self._amp and self._scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = self._model(iris, fingerprint)
                loss = self._criterion(logits, labels)
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            logits = self._model(iris, fingerprint)
            loss = self._criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()

        if self._scheduler is not None:
            self._scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        lr = self._optimizer.param_groups[0]["lr"]
        return {"loss": loss.item(), "accuracy": acc, "lr": lr}

    def _to_device(self, tensor: Any) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self._device, non_blocking=True)
        return torch.tensor(tensor, device=self._device)
