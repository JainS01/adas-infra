"""Pydantic and PyArrow schemas — single source of truth for data shapes."""

from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA, BiometricFrame
from adas_infra.core.schemas.inference import PredictRequest, PredictResponse
from adas_infra.core.schemas.manifest import RunManifest, ShardManifestEntry

__all__ = [
    "BIOMETRIC_ARROW_SCHEMA",
    "BiometricFrame",
    "DeltaOperation",
    "DeltaRecord",
    "PredictRequest",
    "PredictResponse",
    "RunManifest",
    "ShardManifestEntry",
]
