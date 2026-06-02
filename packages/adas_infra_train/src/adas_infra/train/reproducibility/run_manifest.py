"""Build a RunManifest from engine state at the end of training."""

from __future__ import annotations

from typing import Any

from adas_infra.core.determinism.env_snapshot import git_sha, hydra_config_hash
from adas_infra.core.schemas.manifest import RunManifest


def build_run_manifest(
    run_id: str,
    cfg: Any,
    model: Any,
    val_metrics: dict[str, float],
    delta_log_offset: int = 0,
) -> RunManifest:
    """Construct the reproducibility 4-tuple and return a RunManifest.

    Collects git_sha and hydra_config_hash at call time (end of training),
    so the manifest reflects the exact code and config used for the run.
    """
    try:
        cfg_dict = cfg if isinstance(cfg, dict) else dict(cfg) if hasattr(cfg, "keys") else {}
    except Exception:
        cfg_dict = {}

    trainer_cfg = getattr(cfg, "trainer", cfg)
    profile = str(getattr(trainer_cfg, "profile", "local_mock"))
    max_steps = int(getattr(trainer_cfg, "max_steps", 100))
    model_name = str(getattr(getattr(cfg, "model", cfg), "name", "fusion_baseline"))

    num_classes = getattr(model, "num_classes", getattr(model, "_num_classes", 1))

    return RunManifest(
        run_id=run_id,
        git_sha=git_sha(),
        hydra_config_hash=hydra_config_hash(cfg_dict),
        delta_log_offset=delta_log_offset,
        profile=profile,
        model_name=model_name,
        num_classes=int(num_classes),
        max_steps=max_steps,
        best_val_loss=val_metrics.get("val_loss"),
        extra_tags={"val_accuracy": str(val_metrics.get("val_accuracy", ""))},
    )
