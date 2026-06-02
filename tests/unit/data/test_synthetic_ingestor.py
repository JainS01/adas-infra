"""Unit tests for SyntheticIngestor — covers generation, schema, and error handling."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from adas_infra.core.errors import ShardNotFoundError
from adas_infra.data.ingestion.synthetic_ingestor import SyntheticIngestor, SyntheticIngestorConfig
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA


class TestSyntheticIngestor:
    def test_generates_correct_schema(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=5, samples_per_subject=2, seed=1, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        shard_ids = ingestor.generate_shards(num_shards=2)
        table = ingestor.ingest(shard_ids)

        assert table.schema.equals(BIOMETRIC_ARROW_SCHEMA), (
            f"Schema mismatch:\n  got:      {table.schema}\n  expected: {BIOMETRIC_ARROW_SCHEMA}"
        )

    def test_expected_row_count(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=4, samples_per_subject=3, seed=2, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        shard_ids = ingestor.generate_shards(num_shards=2)
        table = ingestor.ingest(shard_ids)
        assert len(table) == 4 * 3

    def test_all_iris_bytes_non_empty(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=3, samples_per_subject=2, seed=3, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        shard_ids = ingestor.generate_shards(num_shards=1)
        table = ingestor.ingest(shard_ids)

        import pyarrow.compute as pc
        null_count = pc.sum(pc.is_null(table.column("iris_bytes"))).as_py()
        assert null_count == 0, f"Found {null_count} null iris_bytes"

    def test_all_fingerprint_bytes_non_empty(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=3, samples_per_subject=2, seed=4, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        shard_ids = ingestor.generate_shards(num_shards=1)
        table = ingestor.ingest(shard_ids)

        import pyarrow.compute as pc
        null_count = pc.sum(pc.is_null(table.column("fingerprint_bytes"))).as_py()
        assert null_count == 0

    def test_splits_present(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=10, samples_per_subject=5, seed=5, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        shard_ids = ingestor.generate_shards(num_shards=2)
        table = ingestor.ingest(shard_ids)

        import pyarrow.compute as pc
        splits = set(pc.unique(table.column("split")).to_pylist())
        assert "train" in splits

    def test_unknown_shard_raises_error(self, tmp_path: Path):
        cfg = SyntheticIngestorConfig(num_subjects=2, samples_per_subject=1, seed=6, output_dir=str(tmp_path))
        ingestor = SyntheticIngestor(cfg=cfg)
        ingestor.generate_shards(num_shards=1)

        with pytest.raises(ShardNotFoundError):
            ingestor.ingest(["nonexistent_shard"])

    def test_deterministic_generation(self, tmp_path: Path):
        """Same seed → same images."""
        cfg1 = SyntheticIngestorConfig(num_subjects=2, samples_per_subject=1, seed=99, output_dir=str(tmp_path / "run1"))
        cfg2 = SyntheticIngestorConfig(num_subjects=2, samples_per_subject=1, seed=99, output_dir=str(tmp_path / "run2"))

        i1 = SyntheticIngestor(cfg=cfg1)
        i2 = SyntheticIngestor(cfg=cfg2)

        ids1 = i1.generate_shards(num_shards=1)
        ids2 = i2.generate_shards(num_shards=1)

        t1 = i1.ingest(ids1)
        t2 = i2.ingest(ids2)

        # Iris bytes of first row should be identical
        assert t1.column("iris_bytes")[0].as_py() == t2.column("iris_bytes")[0].as_py()
