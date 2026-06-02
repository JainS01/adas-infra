"""DeltaLog — append-only write-ahead log for shard-level change events.

The WAL is a newline-delimited JSON file (one DeltaRecord per line).
Each record is checksum-verified on read to detect corruption.  The
log never mutates existing entries — it only appends.

This file-backed WAL is the local-mock equivalent of Azure Event Grid's
durable event stream. The cloud profile would persist records to Postgres
instead, but the DeltaLog class stays unchanged — only the manifest store
changes.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from adas_infra.core.errors import DeltaLogCorruptionError
from adas_infra.core.schemas.delta_record import DeltaRecord

logger = logging.getLogger(__name__)

_WAL_FILENAME = "delta_log.ndjson"


class DeltaLog:
    """Append-only WAL backed by a local NDJSON file.

    Thread-safe: uses a file-level lock for concurrent appends.
    Each call to `append` is a synchronous fsync write.

    Offset semantics: the offset is the number of records written,
    used in the reproducibility 4-tuple (§0.7).
    """

    def __init__(self, wal_dir: Path) -> None:
        wal_dir.mkdir(parents=True, exist_ok=True)
        self._path = wal_dir / _WAL_FILENAME
        self._lock = threading.Lock()
        self._offset = self._count_existing_records()

    def append(self, record: DeltaRecord) -> int:
        """Append one DeltaRecord to stable storage; return the new WAL offset.

        Calls os.fsync() after flush() so the kernel buffer is committed to
        disk before we update the in-memory offset.  This guarantees the WAL
        is crash-consistent: a process killed after append() returns will find
        the record on the next replay().
        """
        line = record.model_dump_json() + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._offset += 1
            return self._offset

    def replay(self, from_offset: int = 0) -> list[DeltaRecord]:
        """Read all records from *from_offset* onward, verifying checksums.

        *from_offset* is a **record count** (not a line number), so blank lines
        and comment lines in the WAL do not affect the offset calculation.

        Raises DeltaLogCorruptionError on the first corrupted record.
        """
        if not self._path.exists():
            return []

        records: list[DeltaRecord] = []
        record_count = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                if record_count < from_offset:
                    record_count += 1
                    continue
                record = DeltaRecord.model_validate_json(raw)
                if not record.verify():
                    raise DeltaLogCorruptionError(
                        f"Checksum mismatch at WAL record {record_count}: shard={record.shard_id}"
                    )
                records.append(record)
                record_count += 1
        return records

    @property
    def offset(self) -> int:
        return self._offset

    def _count_existing_records(self) -> int:
        if not self._path.exists():
            return 0
        with self._path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
