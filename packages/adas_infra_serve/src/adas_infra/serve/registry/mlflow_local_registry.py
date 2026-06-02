"""MLflowLocalRegistry — local file-based model registry (local_mock profile).

Uses MLflow Tracking (file:./mlruns) as the backing store.  This mirrors the
AzureMLRegistry interface exactly; downstream consumers bind to BaseModelRegistry
and never reference either concrete class.

The run_manifest is persisted as MLflow tags on every registered model version
so the (git_sha, config_hash, delta_offset, profile) 4-tuple is always auditable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from adas_infra.core.contracts.model_registry import BaseModelRegistry
from adas_infra.core.errors import ModelRegistryError
from adas_infra.core.schemas.manifest import RunManifest

logger = logging.getLogger(__name__)


class MLflowLocalRegistry(BaseModelRegistry):
    """Stores model artifacts in the local MLflow tracking server (file://./mlruns)."""

    def __init__(self, tracking_uri: str = "file:./mlruns", **kwargs: Any) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._tracking_uri = tracking_uri
        logger.info("MLflowLocalRegistry: tracking_uri=%s", tracking_uri)

    def register(
        self,
        model_path: Path,
        manifest: RunManifest,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Log the model artifact and register it under manifest.model_name.

        Returns the model URI: models:/<name>/<version>
        """
        experiment_name = f"adas-{manifest.model_name}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"run_{manifest.run_id}") as run:
            mlflow_run_id = run.info.run_id

            for key, value in manifest.as_mlflow_tags().items():
                mlflow.set_tag(key, value)

            if metadata:
                mlflow.log_params({k: str(v) for k, v in metadata.items()})

            artifact_path = "model"
            mlflow.log_artifact(str(model_path), artifact_path=artifact_path)

            artifact_uri = f"runs:/{mlflow_run_id}/{artifact_path}"
            model_version = mlflow.register_model(
                model_uri=artifact_uri,
                name=manifest.model_name,
                tags=manifest.as_mlflow_tags(),
            )

        version_uri = f"models:/{manifest.model_name}/{model_version.version}"
        logger.info(
            "MLflowLocalRegistry: registered %s as version %s",
            manifest.model_name,
            model_version.version,
        )
        return version_uri

    def fetch(self, version_uri: str, dest: Path) -> Path:
        """Download the model artifact from the local mlruns directory."""
        dest.mkdir(parents=True, exist_ok=True)
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=version_uri, dst_path=str(dest)
            )
        except Exception as exc:
            raise ModelRegistryError(f"Cannot fetch {version_uri}: {exc}") from exc
        return Path(local_path)

    def latest_uri(self, model_name: str) -> str:
        try:
            versions = self._client.get_latest_versions(model_name)
            if not versions:
                raise ModelRegistryError(f"No versions registered for '{model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            return f"models:/{model_name}/{latest.version}"
        except mlflow.exceptions.MlflowException as exc:
            raise ModelRegistryError(str(exc)) from exc

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")
        except mlflow.exceptions.MlflowException as exc:
            raise ModelRegistryError(str(exc)) from exc
        return [
            {
                "version": v.version,
                "run_id": v.run_id,
                "status": v.status,
                "tags": dict(v.tags or {}),
            }
            for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
        ]
