"""SingleDeviceEngine — CPU or single-GPU training engine.

This is the judge / local_mock path.  It implements the full BaseTrainer
template contract without touching distributed collectives, so the entire
pipeline can be exercised in a CI container with no GPU and no NCCL.

The engine:
  1. Resolves the device (cuda:0 or cpu)
  2. Builds the FusionBaseline model from config
  3. Materialises Plasma ObjectRefs via RayDatasetLoader → PlasmaPrefetcher
  4. Runs TrainLoop for max_steps
  5. Runs EvalLoop on the validation split
  6. Fires all registered hooks at the end (incl. RunManifestHook, CheckpointHook)
  7. Returns a RunManifest
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim

from adas_infra.core.contracts.trainer import BaseTrainer
from adas_infra.core.determinism.env_snapshot import git_sha, hydra_config_hash
from adas_infra.core.determinism.seeding import seed_everything
from adas_infra.core.schemas.manifest import RunManifest
from adas_infra.train.loops.eval_loop import EvalLoop
from adas_infra.train.loops.hooks import LoggingHook, RunManifestHook, TrainingHook
from adas_infra.train.loops.train_loop import TrainLoop
from adas_infra.train.models.fusion_baseline import FusionBaseline
from adas_infra.train.optim.schedulers import build_scheduler
from adas_infra.train.reproducibility.run_manifest import build_run_manifest

logger = logging.getLogger(__name__)


class SingleDeviceEngine(BaseTrainer):
    """Train on a single CPU or GPU; full BaseTrainer template contract.

    Config keys consumed (Hydra DictConfig or plain dict):
      trainer.seed          int   (default 42)
      trainer.max_steps     int   (default 100)
      trainer.batch_size    int   (default 32)
      trainer.lr            float (default 1e-3)
      trainer.weight_decay  float (default 1e-4)
      trainer.checkpoint_dir str  (default ./checkpoints)
      trainer.profile       str   (default local_mock)
      model.num_classes     int   (default 20)
      model.iris_embed_dim  int   (default 128)
      model.fp_embed_dim    int   (default 128)
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._device: torch.device | None = None
        self._model: FusionBaseline | None = None
        self._optimizer: optim.Optimizer | None = None
        self._scheduler: Any = None
        self._run_id: str = str(uuid.uuid4())[:8]

    # ── BaseTrainer template methods ──────────────────────────────────────────

    @staticmethod
    def _get(cfg: Any, *keys: str, default: Any) -> Any:
        """Walk a dotted attribute chain, returning *default* if any key is absent.

        Example: _get(cfg, "trainer", "seed", default=42)
        """
        obj = cfg
        for key in keys:
            obj = getattr(obj, key, None)
            if obj is None:
                return default
        return obj

    def _setup(self, model: Any, train_data: Any) -> None:
        cfg = self._cfg
        g = self._get  # shorter alias for repeated calls

        seed = int(g(cfg, "trainer", "seed", default=42))
        seed_everything(seed)

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        logger.info("SingleDeviceEngine: device=%s", self._device)

        if isinstance(model, FusionBaseline):
            self._model = model.to(self._device)
        else:
            self._model = FusionBaseline(
                num_classes=int(g(cfg, "model", "num_classes", default=20)),
                iris_embed_dim=int(g(cfg, "model", "iris_embed_dim", default=128)),
                fp_embed_dim=int(g(cfg, "model", "fp_embed_dim", default=128)),
            ).to(self._device)

        max_steps = int(g(cfg, "trainer", "max_steps", default=100))
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=float(g(cfg, "trainer", "lr", default=1e-3)),
            weight_decay=float(g(cfg, "trainer", "weight_decay", default=1e-4)),
        )
        self._scheduler = build_scheduler(self._optimizer, max_steps)

        ckpt_dir = Path(str(g(cfg, "trainer", "checkpoint_dir", default="./checkpoints")))
        self.register_hook(LoggingHook(log_every_n_steps=10))
        self.register_hook(RunManifestHook(checkpoint_dir=ckpt_dir, mlflow_run_id=None))

        logger.info(
            "SingleDeviceEngine: model=%s params=%d device=%s",
            type(self._model).__name__,
            sum(p.numel() for p in self._model.parameters()),
            self._device,
        )

    def _run_loop(
        self,
        model: Any,
        train_data: Any,
        val_data: Any | None,
    ) -> RunManifest:
        assert self._model is not None
        assert self._optimizer is not None
        assert self._device is not None

        max_steps = int(self._get(self._cfg, "trainer", "max_steps", default=100))

        train_loop = TrainLoop(
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            device=self._device,
            max_steps=max_steps,
            hooks=[h for h in self._hooks if isinstance(h, TrainingHook)],
            amp=False,
        )
        final_metrics = train_loop.run(train_data)

        val_metrics: dict[str, float] = {}
        if val_data is not None:
            eval_loop = EvalLoop(model=self._model, device=self._device)
            val_metrics = eval_loop.run(val_data)
            logger.info(
                "Validation — loss=%.4f acc=%.4f",
                val_metrics.get("val_loss", 0.0),
                val_metrics.get("val_accuracy", 0.0),
            )

        manifest = build_run_manifest(
            run_id=self._run_id,
            cfg=cfg,
            model=self._model,
            val_metrics=val_metrics,
        )
        return manifest

    def _teardown(self, model: Any, manifest: RunManifest) -> None:
        for hook in self._hooks:
            hook.on_train_end(manifest)
        logger.info("SingleDeviceEngine: run %s complete", self._run_id)

    @property
    def model(self) -> FusionBaseline | None:
        return self._model
