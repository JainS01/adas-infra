"""RunManifest and ShardManifestEntry — reproducibility and bookkeeping schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from adas_infra.core.schemas._versioning import versioned_schema

CURRENT_VERSION = 1


class ShardManifestEntry(BaseModel):
    """Tracks the lifecycle of a single data shard."""

    shard_id: str
    path: str
    byte_size: int = Field(ge=0)
    num_rows: int = Field(ge=0)
    ingested: bool = False
    ingested_at: datetime | None = None


@versioned_schema(current=CURRENT_VERSION)
class RunManifest(BaseModel):
    """The reproducibility 4-tuple persisted as MLflow tags AND run_manifest.json.

    Every training run MUST record this tuple so results are attributable to a
    precise (code, config, data, environment) quadrant.
    """

    schema_version: int = Field(default=CURRENT_VERSION)
    run_id: str
    git_sha: str
    hydra_config_hash: str
    delta_log_offset: int = Field(ge=0)
    profile: str
    model_name: str
    num_classes: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    best_val_loss: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra_tags: dict[str, str] = Field(default_factory=dict)

    def as_mlflow_tags(self) -> dict[str, str]:
        """Flatten the manifest into a flat string dict suitable for MLflow tags."""
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "hydra_config_hash": self.hydra_config_hash,
            "delta_log_offset": str(self.delta_log_offset),
            "profile": self.profile,
            "model_name": self.model_name,
            "schema_version": str(self.schema_version),
            **self.extra_tags,
        }
