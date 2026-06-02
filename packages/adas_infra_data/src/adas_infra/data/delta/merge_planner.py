"""MergePlanner — decides which shards to hand to the ingestor each run.

The planner is a pure function on the manifest store state; it emits
a deterministic shard list without touching storage directly.  This
decoupling means the training engine can call `plan()` before binding
the ingestor, enabling dry-run mode and unit testing without I/O.

Strategies
----------
incremental
    Only shards marked pending (not yet ingested) in the manifest store.
    Default for streaming / production use.
full_scan
    Every shard known to the manifest store, regardless of ingestion status.
    Use for reprocessing or initial cold-start.
"""

from __future__ import annotations

import logging
from enum import Enum

from adas_infra.core.contracts.manifest_store import BaseManifestStore

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    INCREMENTAL = "incremental"
    FULL_SCAN = "full_scan"


class MergePlanner:
    """Computes the shard list for the next ingestion run.

    The planner never modifies the manifest store — it only reads it.
    Marking shards as ingested is the ingestor's responsibility, called
    explicitly after successful Arrow table construction.
    """

    def __init__(
        self,
        manifest_store: BaseManifestStore,
        strategy: MergeStrategy = MergeStrategy.INCREMENTAL,
    ) -> None:
        self._store = manifest_store
        self._strategy = strategy

    def plan(self) -> list[str]:
        """Return an ordered list of shard IDs to ingest in the next run."""
        if self._strategy == MergeStrategy.FULL_SCAN:
            # Re-mark all shards as pending so they are included
            all_entries = self._store.get_pending_shards()
            shard_ids = [e.shard_id for e in all_entries]
        else:
            pending = self._store.get_pending_shards()
            shard_ids = [e.shard_id for e in pending]

        logger.info(
            "MergePlanner(%s): %d shards selected for ingestion",
            self._strategy.value,
            len(shard_ids),
        )
        return shard_ids

    def commit(self, ingested_shard_ids: list[str]) -> None:
        """Mark the given shards as successfully ingested in the manifest."""
        self._store.mark_ingested(ingested_shard_ids)
        logger.debug("MergePlanner: committed %d shards", len(ingested_shard_ids))
