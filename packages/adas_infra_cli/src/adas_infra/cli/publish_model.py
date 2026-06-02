"""adas-publish-model — register the latest checkpoint with the model registry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a trained model to the registry")
    parser.add_argument("--run-id", default=None, help="Run ID; uses .last_run_id if omitted")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt file")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--model-name", default="fusion_baseline")
    args = parser.parse_args()

    from adas_infra.serve.registry.mlflow_local_registry import MLflowLocalRegistry
    from adas_infra.core.schemas.manifest import RunManifest
    from adas_infra.core.determinism.env_snapshot import git_sha

    run_id = args.run_id
    if run_id is None or run_id == "latest":
        last_id_file = Path(".last_run_id")
        run_id = last_id_file.read_text().strip() if last_id_file.exists() else "manual"

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        last_path_file = Path(".last_model_path")
        ckpt_path = last_path_file.read_text().strip() if last_path_file.exists() else "./checkpoints/fusion.pt"

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run 'adas-train' first.")

    manifest = RunManifest(
        run_id=run_id,
        git_sha=git_sha(),
        hydra_config_hash="manual",
        delta_log_offset=0,
        profile="local_mock",
        model_name=args.model_name,
        num_classes=20,
        max_steps=50,
    )

    reg = MLflowLocalRegistry(tracking_uri=args.tracking_uri)
    version_uri = reg.register(ckpt, manifest)
    log.info("Published: %s", version_uri)
    print(version_uri)
