"""MLflowLocalRegistry — local model registry (local_mock profile).

Uses MLflow Tracking (sqlite:///mlflow.db) as the backing store.  The filesystem
backend is in maintenance mode and rejected by MLflow >=3.x, and the model
registry has always required a database backend, so SQLite is used.  This mirrors
the AzureMLRegistry interface exactly; downstream consumers bind to
BaseModelRegistry and never reference either concrete class.

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


class MLflowLocalRegistry(BaseModelRegistry):  # type: ignore[misc]
    """Stores model artifacts in the local MLflow tracking store (sqlite:///mlflow.db)."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", **kwargs: Any) -> None:
        # A SQLite backend defaults artifacts to ./mlruns under the cwd; co-locate
        # them with the DB instead so the working tree stays clean. Ensure the
        # parent dir exists so SQLite can create the file on first connection.
        self._artifact_root: str | None = None
        if tracking_uri.startswith("sqlite:///"):
            db_path = Path(tracking_uri.removeprefix("sqlite:///"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._artifact_root = (db_path.parent / "artifacts").as_uri()
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
        from adas_infra.serve.export.torchscript_exporter import TorchScriptExporter

        experiment_name = f"adas-{manifest.model_name}"
        if self._artifact_root and self._client.get_experiment_by_name(experiment_name) is None:
            self._client.create_experiment(experiment_name, artifact_location=self._artifact_root)
        mlflow.set_experiment(experiment_name)
        tags = manifest.as_mlflow_tags()

        with mlflow.start_run(run_name=f"run_{manifest.run_id}"):
            for key, value in tags.items():
                mlflow.set_tag(key, value)

            if metadata:
                mlflow.log_params({k: str(v) for k, v in metadata.items()})

            # MLflow 3.x registers from a first-class *logged model*, not a raw
            # artifact, so load the TorchScript archive and log it as one.
            scripted = TorchScriptExporter.load(model_path)
            model_info = mlflow.pytorch.log_model(scripted, name="model")

            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=manifest.model_name,
                tags=tags,
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
