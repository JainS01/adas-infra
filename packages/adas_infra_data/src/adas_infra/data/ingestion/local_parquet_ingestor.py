"""LocalParquetIngestor — reads Hive-partitioned Parquet shards from a local filesystem.

Mirrors AzureBlobIngestor but reads from the local NVME path (judge / local_mock profile).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from adas_infra.core.errors import ShardNotFoundError
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA

logger = logging.getLogger(__name__)


class LocalParquetIngestor:
    """Reads Parquet shard files from a local directory tree.

    Expected layout::

        <storage_root>/
            shard_0000_<hash>.parquet
            shard_0001_<hash>.parquet
            ...

    Shard IDs are the file stems without the .parquet extension.
    """

    def __init__(self, storage_root: str, **kwargs: Any) -> None:
        self._root = Path(storage_root)
        if not self._root.exists():
            raise ShardNotFoundError("(root)", str(self._root))

    def list_shards(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.parquet"))

    def ingest(self, shard_ids: list[str]) -> pa.Table:
        tables: list[pa.Table] = []
        for sid in shard_ids:
            path = self._root / f"{sid}.parquet"
            if not path.exists():
                raise ShardNotFoundError(shard_id=sid, location=str(self._root))
            tables.append(pq.read_table(path, schema=BIOMETRIC_ARROW_SCHEMA))  # type: ignore[no-untyped-call]
            logger.debug("Loaded shard %s (%d rows)", sid, len(tables[-1]))
        if not tables:
            return pa.table(
                {f.name: [] for f in BIOMETRIC_ARROW_SCHEMA},
                schema=BIOMETRIC_ARROW_SCHEMA,
            )
        return pa.concat_tables(tables)
