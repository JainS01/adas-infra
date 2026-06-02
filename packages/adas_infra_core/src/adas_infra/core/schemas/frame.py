"""BiometricFrame — schema for a single multimodal biometric sample."""

from __future__ import annotations

import pyarrow as pa
from pydantic import BaseModel, Field

from adas_infra.core.schemas._versioning import versioned_schema

CURRENT_VERSION = 1

# PyArrow schema — used for zero-copy column slices in the data plane.
# Readers MUST accept schema_version <= CURRENT_VERSION.
BIOMETRIC_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("schema_version", pa.int32()),
        pa.field("subject_id", pa.string()),
        pa.field("sample_id", pa.string()),
        pa.field("iris_bytes", pa.binary()),       # raw image bytes (JPEG / PNG)
        pa.field("fingerprint_bytes", pa.binary()),
        pa.field("label", pa.int64()),             # subject ordinal for classification
        pa.field("split", pa.string()),            # "train" | "val" | "test"
        pa.field("source_shard", pa.string()),     # shard_id that provided this row
    ],
    metadata={"schema_name": "BiometricFrame", "schema_version": str(CURRENT_VERSION)},
)


@versioned_schema(current=CURRENT_VERSION)
class BiometricFrame(BaseModel):
    """Pydantic model mirroring BIOMETRIC_ARROW_SCHEMA for API / validation use."""

    schema_version: int = Field(default=CURRENT_VERSION, ge=1)
    subject_id: str
    sample_id: str
    iris_bytes: bytes
    fingerprint_bytes: bytes
    label: int = Field(ge=0)
    split: str = Field(default="train", pattern=r"^(train|val|test)$")
    source_shard: str = Field(default="")
