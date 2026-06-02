"""Ingestor protocol — reads shards and returns an Arrow table."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class Ingestor(Protocol):
    """Reads one or more shard IDs from the underlying storage and returns an Arrow table.

    Implementations: AzureBlobIngestor, LocalParquetIngestor, SyntheticIngestor.
    The merge planner supplies the shard_ids list; the ingestor never queries the delta log.
    """

    def ingest(self, shard_ids: list[str]) -> pa.Table:
        """Return a PyArrow Table covering the requested shards.

        Columns (minimum required):
          subject_id   : string
          sample_id    : string
          iris_bytes   : binary  (raw image bytes, any format)
          fingerprint_bytes : binary
          label        : int64   (subject ordinal for classification)

        Raises:
            MissingModalityError: If any sample is missing a required modality.
            ShardNotFoundError:   If a shard_id cannot be resolved.
        """
        ...

    def list_shards(self) -> list[str]:
        """Return all shard IDs available in the backing store."""
        ...
