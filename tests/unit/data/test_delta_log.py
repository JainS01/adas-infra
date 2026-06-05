"""Unit tests for DeltaLog WAL — append, replay, corruption detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from adas_infra.core.errors import DeltaLogCorruptionError
from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
from adas_infra.data.delta.delta_log import DeltaLog


class TestDeltaLog:
    def test_append_increments_offset(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        assert wal.offset == 0
        r1 = DeltaRecord(
            shard_id="s1",
            operation=DeltaOperation.ADD,
            path="p1",
            byte_size=10,
            num_rows=5,
        )
        wal.append(r1)
        assert wal.offset == 1
        wal.append(r1)
        assert wal.offset == 2

    def test_replay_returns_all_records(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        records = [
            DeltaRecord(
                shard_id=f"s{i}",
                operation=DeltaOperation.ADD,
                path=f"p{i}",
                byte_size=i,
                num_rows=i,
            )
            for i in range(5)
        ]
        for r in records:
            wal.append(r)
        replayed = wal.replay()
        assert len(replayed) == 5
        assert [r.shard_id for r in replayed] == [f"s{i}" for i in range(5)]

    def test_replay_from_offset(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        for i in range(6):
            wal.append(
                DeltaRecord(
                    shard_id=f"s{i}",
                    operation=DeltaOperation.ADD,
                    path=f"p{i}",
                    byte_size=0,
                    num_rows=0,
                )
            )
        tail = wal.replay(from_offset=3)
        assert len(tail) == 3
        assert tail[0].shard_id == "s3"

    def test_corruption_raises_error(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        r = DeltaRecord(
            shard_id="s1",
            operation=DeltaOperation.ADD,
            path="p1",
            byte_size=0,
            num_rows=0,
        )
        wal.append(r)

        # Corrupt the WAL file
        wal_file = tmp_path / "delta_log.ndjson"
        content = wal_file.read_text()
        corrupted = content.replace(r.checksum, "badc0ffe" * 8)
        wal_file.write_text(corrupted)

        with pytest.raises(DeltaLogCorruptionError):
            wal.replay()

    def test_persists_across_instances(self, tmp_path: Path):
        """Offset is recovered by counting lines on re-open."""
        wal1 = DeltaLog(wal_dir=tmp_path)
        for i in range(3):
            wal1.append(
                DeltaRecord(
                    shard_id=f"s{i}",
                    operation=DeltaOperation.ADD,
                    path="p",
                    byte_size=0,
                    num_rows=0,
                )
            )

        wal2 = DeltaLog(wal_dir=tmp_path)
        assert wal2.offset == 3
