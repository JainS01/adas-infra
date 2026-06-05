"""Build a RunManifest from engine state at the end of training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adas_infra.core.determinism.env_snapshot import git_sha, hydra_config_hash
from adas_infra.core.schemas.manifest import RunManifest


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Recursively materialise any config object into a plain, JSON-hashable dict.

    Handles the three shapes the engine sees:
      - OmegaConf DictConfig (Hydra CLI path) — resolved to a container so the
        config hash reflects interpolated values, not ``${...}`` placeholders.
      - Mapping / dict (programmatic callers).
      - Plain classes or instances (the Hydra-free ``pipeline.py`` smoke path,
        whose ``_Cfg`` carries nested ``trainer`` / ``model`` classes).

    Without this, the non-Hydra path previously hashed an empty ``{}``, leaving
    the reproducibility tuple's config component nominal.
    """
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass

    if isinstance(cfg, Mapping):
        return {str(k): _to_plain_value(v) for k, v in cfg.items()}

    # Plain class or instance: collect public attributes (skip dunders/callables).
    out: dict[str, Any] = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        val = getattr(cfg, key, None)
        if callable(val):
            continue
        out[key] = _to_plain_value(val)
    return out


def _to_plain_value(val: Any) -> Any:
    """Recursively normalise a single config value into a JSON-serialisable form."""
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, Mapping):
        return {str(k): _to_plain_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_plain_value(v) for v in val]
    # Nested config class/instance (e.g. _Cfg.trainer) — recurse.
    if hasattr(val, "__dict__") or isinstance(val, type):
        return _to_plain_dict(val)
    return str(val)


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

    *delta_log_offset* is the WAL position at ingestion time — the data-version
    axis of the reproducibility tuple. Callers must pass the live manifest-store
    offset; a 0 here means "no delta provenance recorded".
    """
    cfg_dict = _to_plain_dict(cfg)

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
