"""BaseModelRegistry ABC — versioned model artifact lifecycle."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from adas_infra.core.schemas.manifest import RunManifest


class BaseModelRegistry(abc.ABC):
    """Push, fetch, and promote versioned model artifacts.

    Implementations: MLflowLocalRegistry (local), AzureMLRegistry (cloud).
    Both expose an identical interface; the training engine only talks to this ABC.
    """

    @abc.abstractmethod
    def register(
        self,
        model_path: Path,
        manifest: RunManifest,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist the model artifact and return a version URI.

        The run_manifest is stored as registry tags so reproducibility is auditable.
        """

    @abc.abstractmethod
    def fetch(self, version_uri: str, dest: Path) -> Path:
        """Download the model artifact to *dest* and return the local path."""

    @abc.abstractmethod
    def latest_uri(self, model_name: str) -> str:
        """Return the URI of the most recently registered version."""

    @abc.abstractmethod
    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        """Return version metadata dicts in descending registration order."""
