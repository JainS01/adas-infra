"""ManifestStoreSQLite — local SQLite implementation of BaseManifestStore.

Used in the local_mock profile. The Postgres implementation (cloud profile)
exposes the identical BaseManifestStore ABC; the merge planner and training
engine never reference either concrete class.

Schema:
    shards(shard_id TEXT PK, path TEXT, byte_size INT, num_rows INT,
           ingested INT DEFAULT 0, ingested_at TEXT)
    wal_meta(key TEXT PK, value TEXT)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adas_infra.core.contracts.manifest_store import BaseManifestStore
from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
from adas_infra.core.schemas.manifest import ShardManifestEntry

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id    TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    byte_size   INTEGER NOT NULL DEFAULT 0,
    num_rows    INTEGER NOT NULL DEFAULT 0,
    ingested    INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS wal_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO wal_meta (key, value) VALUES ('wal_offset', '0');
"""


class ManifestStoreSQLite(BaseManifestStore):  # type: ignore[misc]
    """Thread-safe SQLite manifest store for the local_mock profile."""

    def __init__(self, db_path: str = "./state/manifest.db", **kwargs: object) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False allows cross-thread use; _lock serialises writers.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info("ManifestStoreSQLite: opened %s", path)

    def record_delta(self, record: DeltaRecord) -> None:
        """Upsert the shard entry and bump the WAL offset counter."""
        with self._lock, self._conn:
            if record.operation == DeltaOperation.REMOVE:
                self._conn.execute("DELETE FROM shards WHERE shard_id = ?", (record.shard_id,))
            else:
                self._conn.execute(
                    """
                    INSERT INTO shards (shard_id, path, byte_size, num_rows)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(shard_id) DO UPDATE SET
                        path=excluded.path,
                        byte_size=excluded.byte_size,
                        num_rows=excluded.num_rows,
                        ingested=0,
                        ingested_at=NULL
                    """,
                    (record.shard_id, record.path, record.byte_size, record.num_rows),
                )
            self._conn.execute(
                "UPDATE wal_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
                "WHERE key = 'wal_offset'"
            )

    def get_pending_shards(self) -> list[ShardManifestEntry]:
        rows = self._conn.execute(
            "SELECT shard_id, path, byte_size, num_rows, ingested, ingested_at "
            "FROM shards WHERE ingested = 0"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_all_shards(self) -> list[ShardManifestEntry]:
        rows = self._conn.execute(
            "SELECT shard_id, path, byte_size, num_rows, ingested, ingested_at FROM shards"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(r: tuple[Any, ...]) -> ShardManifestEntry:
        return ShardManifestEntry(
            shard_id=r[0],
            path=r[1],
            byte_size=r[2],
            num_rows=r[3],
            ingested=bool(r[4]),
            ingested_at=datetime.fromisoformat(r[5]) if r[5] else None,
        )

    def mark_ingested(self, shard_ids: list[str]) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE shards SET ingested=1, ingested_at=? WHERE shard_id=?",
                [(now, sid) for sid in shard_ids],
            )

    def get_wal_offset(self) -> int:
        row = self._conn.execute("SELECT value FROM wal_meta WHERE key='wal_offset'").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
        logger.info("ManifestStoreSQLite: closed")
