"""BaseManifestStore ABC — transactional shard-delta bookkeeping."""

from __future__ import annotations

import abc

from adas_infra.core.schemas.delta_record import DeltaRecord
from adas_infra.core.schemas.manifest import ShardManifestEntry


class BaseManifestStore(abc.ABC):
    """Tracks which shards exist, which are pending ingestion, and the WAL offset.

    The store owns transactional state; two implementations exist behind this contract:
      - ManifestStoreSQLite  (local profile — ./state/manifest.db)
      - ManifestStorePostgres (cloud profile — Azure Postgres Flexible Server)
    """

    @abc.abstractmethod
    def record_delta(self, record: DeltaRecord) -> None:
        """Append a DeltaRecord to the WAL and mark the shard as pending."""

    @abc.abstractmethod
    def get_pending_shards(self) -> list[ShardManifestEntry]:
        """Return all shards not yet confirmed as ingested in the current epoch."""

    @abc.abstractmethod
    def mark_ingested(self, shard_ids: list[str]) -> None:
        """Atomically mark a list of shards as successfully ingested."""

    @abc.abstractmethod
    def get_wal_offset(self) -> int:
        """Return the current WAL position; used in the reproducibility 4-tuple."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush pending writes and release the database connection."""
