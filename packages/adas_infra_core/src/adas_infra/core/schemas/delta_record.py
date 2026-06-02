"""DeltaRecord — WAL entry emitted by CDC sources."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from adas_infra.core.schemas._versioning import versioned_schema

CURRENT_VERSION = 1


class DeltaOperation(str, Enum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    UPDATE = "UPDATE"


@versioned_schema(current=CURRENT_VERSION)
class DeltaRecord(BaseModel):
    """Immutable WAL entry describing a shard-level change event.

    The checksum field is computed over (shard_id + operation + path) so the
    WAL can be replayed and each record verified for corruption.
    """

    schema_version: int = Field(default=CURRENT_VERSION, ge=1)
    shard_id: str
    operation: DeltaOperation
    path: str
    byte_size: int = Field(ge=0)
    num_rows: int = Field(ge=0)
    checksum: str = Field(default="")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _compute_checksum(self) -> "DeltaRecord":
        if not self.checksum:
            payload = f"{self.shard_id}:{self.operation}:{self.path}".encode()
            self.checksum = hashlib.sha256(payload).hexdigest()
        return self

    def verify(self) -> bool:
        """Return True if the stored checksum matches the record contents."""
        payload = f"{self.shard_id}:{self.operation}:{self.path}".encode()
        return self.checksum == hashlib.sha256(payload).hexdigest()
