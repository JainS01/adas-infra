"""Domain-specific exception hierarchy for the ADAS infra platform."""

from __future__ import annotations


class AdasInfraError(Exception):
    """Base class for all platform errors."""


# ── Data plane ────────────────────────────────────────────────────────────────


class IngestionError(AdasInfraError):
    """Raised when an ingestor cannot read or decode a shard."""


class MissingModalityError(IngestionError):
    """Raised when a required modality file is absent for a sample.

    Example: iris image exists but no matching fingerprint for subject_id.
    """

    def __init__(self, subject_id: str, modality: str, path: str) -> None:
        self.subject_id = subject_id
        self.modality = modality
        self.path = path
        super().__init__(
            f"Missing {modality} for subject '{subject_id}' — expected path: {path}"
        )


class ShardNotFoundError(IngestionError):
    """Raised when a shard referenced by the delta log no longer exists."""

    def __init__(self, shard_id: str, location: str) -> None:
        self.shard_id = shard_id
        self.location = location
        super().__init__(f"Shard '{shard_id}' not found at '{location}'")


class SchemaVersionError(AdasInfraError):
    """Raised when a schema version is newer than the reader supports."""

    def __init__(self, found: int, max_supported: int, schema_name: str) -> None:
        super().__init__(
            f"Schema '{schema_name}' version {found} exceeds reader max {max_supported}. "
            "Upgrade the adas-infra-core package."
        )


class DeltaLogCorruptionError(AdasInfraError):
    """Raised when the WAL checksum does not match the stored record."""


# ── Training plane ────────────────────────────────────────────────────────────


class CheckpointError(AdasInfraError):
    """Raised when checkpoint save or resume fails."""


class ReproducibilityError(AdasInfraError):
    """Raised when run-manifest cannot be persisted (aborts the run)."""


# ── Serve plane ───────────────────────────────────────────────────────────────


class ModelRegistryError(AdasInfraError):
    """Raised on registry push, fetch, or version-lookup failures."""


class InferenceError(AdasInfraError):
    """Raised when the inference endpoint cannot score a request."""


# ── Config plane ──────────────────────────────────────────────────────────────


class ConfigurationError(AdasInfraError):
    """Raised when Hydra config is structurally invalid for the active profile."""
