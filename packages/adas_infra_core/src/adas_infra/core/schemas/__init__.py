"""Pydantic and PyArrow schemas — single source of truth for data shapes."""

from adas_infra.core.schemas.delta_record import DeltaRecord, DeltaOperation
from adas_infra.core.schemas.frame import BiometricFrame, BIOMETRIC_ARROW_SCHEMA
from adas_infra.core.schemas.inference import PredictRequest, PredictResponse
from adas_infra.core.schemas.manifest import RunManifest, ShardManifestEntry

__all__ = [
    "DeltaRecord",
    "DeltaOperation",
    "BiometricFrame",
    "BIOMETRIC_ARROW_SCHEMA",
    "PredictRequest",
    "PredictResponse",
    "RunManifest",
    "ShardManifestEntry",
]
