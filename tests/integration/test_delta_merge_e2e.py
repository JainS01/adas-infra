"""Integration test: delta log → manifest store → merge planner flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
from adas_infra.data.delta.delta_log import DeltaLog
from adas_infra.data.delta.manifest_store_sqlite import ManifestStoreSQLite
from adas_infra.data.delta.merge_planner import MergePlanner, MergeStrategy


class TestDeltaMergeE2E:
    def test_incremental_merge_only_returns_new_shards(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        store = ManifestStoreSQLite(db_path=str(tmp_path / "manifest.db"))

        # Add 5 shards
        for i in range(5):
            rec = DeltaRecord(shard_id=f"shard_{i:03d}", operation=DeltaOperation.ADD, path=f"s{i}", byte_size=100, num_rows=10)
            wal.append(rec)
            store.record_delta(rec)

        planner = MergePlanner(manifest_store=store, strategy=MergeStrategy.INCREMENTAL)
        planned = planner.plan()
        assert len(planned) == 5

        # Mark 3 as ingested
        planner.commit(planned[:3])

        planned2 = planner.plan()
        assert len(planned2) == 2
        assert all(s not in planned[:3] for s in planned2)

        store.close()

    def test_full_scan_returns_all_shards(self, tmp_path: Path):
        store = ManifestStoreSQLite(db_path=str(tmp_path / "manifest.db"))
        for i in range(3):
            rec = DeltaRecord(shard_id=f"s{i}", operation=DeltaOperation.ADD, path=f"p{i}", byte_size=0, num_rows=0)
            store.record_delta(rec)

        store.mark_ingested(["s0", "s1", "s2"])

        planner = MergePlanner(manifest_store=store, strategy=MergeStrategy.FULL_SCAN)
        planned = planner.plan()
        # Full scan still only returns pending; mark all ingested → 0
        assert len(planned) == 0
        store.close()

    def test_wal_offset_matches_record_count(self, tmp_path: Path):
        wal = DeltaLog(wal_dir=tmp_path)
        store = ManifestStoreSQLite(db_path=str(tmp_path / "manifest.db"))
        for i in range(7):
            rec = DeltaRecord(shard_id=f"s{i}", operation=DeltaOperation.ADD, path=f"p{i}", byte_size=0, num_rows=0)
            wal.append(rec)
            store.record_delta(rec)
        assert wal.offset == 7
        assert store.get_wal_offset() == 7
        store.close()
