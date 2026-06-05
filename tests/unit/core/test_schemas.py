"""Unit tests for core schemas: DeltaRecord, RunManifest, PredictRequest."""

from __future__ import annotations

import base64

import pytest

from adas_infra.core.errors import SchemaVersionError
from adas_infra.core.schemas._versioning import (
    read_versioned,
    require_version_lte,
    upcast_to_current,
)
from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
from adas_infra.core.schemas.inference import PredictRequest
from adas_infra.core.schemas.manifest import RunManifest


class TestDeltaRecord:
    def test_checksum_computed_on_construction(self):
        rec = DeltaRecord(
            shard_id="s1",
            operation=DeltaOperation.ADD,
            path="/data/s1.parquet",
            byte_size=1024,
            num_rows=100,
        )
        assert len(rec.checksum) == 64  # SHA-256 hex

    def test_verify_passes_for_valid_record(self):
        rec = DeltaRecord(
            shard_id="s1",
            operation=DeltaOperation.ADD,
            path="/data/s1.parquet",
            byte_size=0,
            num_rows=0,
        )
        assert rec.verify() is True

    def test_verify_fails_if_checksum_tampered(self):
        rec = DeltaRecord(
            shard_id="s1",
            operation=DeltaOperation.ADD,
            path="/data/s1.parquet",
            byte_size=0,
            num_rows=0,
        )
        rec.checksum = "deadbeef" * 8
        assert rec.verify() is False

    def test_roundtrip_json(self):
        rec = DeltaRecord(
            shard_id="s2",
            operation=DeltaOperation.REMOVE,
            path="/d/s2.parquet",
            byte_size=512,
            num_rows=50,
        )
        restored = DeltaRecord.model_validate_json(rec.model_dump_json())
        assert restored.shard_id == "s2"
        assert restored.operation == DeltaOperation.REMOVE


class TestReadVersioned:
    """The reader upcast path used by DeltaLog.replay."""

    def test_reads_current_version(self):
        data = {
            "schema_version": 1,
            "shard_id": "s1",
            "operation": "ADD",
            "path": "/d/s1.parquet",
            "byte_size": 0,
            "num_rows": 0,
        }
        rec = read_versioned(DeltaRecord, data)
        assert rec.shard_id == "s1"

    def test_rejects_future_version(self):
        data = {
            "schema_version": 99,
            "shard_id": "s1",
            "operation": "ADD",
            "path": "/d/s1.parquet",
            "byte_size": 0,
            "num_rows": 0,
        }
        with pytest.raises(SchemaVersionError):
            read_versioned(DeltaRecord, data)

    def test_upcast_applies_migration_chain(self):
        def _v1_to_v2(d: dict) -> dict:
            d = {**d, "schema_version": 2, "added": True}
            return d

        out = upcast_to_current({"schema_version": 1}, {1: _v1_to_v2})
        assert out["schema_version"] == 2
        assert out["added"] is True


class TestRunManifest:
    def test_as_mlflow_tags_returns_flat_dict(self):
        m = RunManifest(
            run_id="abc",
            git_sha="deadcafe",
            hydra_config_hash="0000",
            delta_log_offset=5,
            profile="local_mock",
            model_name="fusion_baseline",
            num_classes=20,
            max_steps=50,
        )
        tags = m.as_mlflow_tags()
        assert tags["git_sha"] == "deadcafe"
        assert tags["profile"] == "local_mock"
        assert isinstance(tags["delta_log_offset"], str)

    def test_schema_version_lte_passes(self):
        require_version_lte(found=1, maximum=1, schema_name="test")

    def test_schema_version_lte_fails(self):
        with pytest.raises(SchemaVersionError):
            require_version_lte(found=2, maximum=1, schema_name="test")


class TestPredictRequest:
    def _make_request(self):
        raw = b"\x89PNG" + b"\x00" * 100  # fake PNG bytes
        b64 = base64.b64encode(raw).decode()
        return PredictRequest(request_id="req-001", iris_b64=b64, fingerprint_b64=b64, top_k=3)

    def test_construction(self):
        req = self._make_request()
        assert req.request_id == "req-001"
        assert req.top_k == 3

    def test_iris_bytes_roundtrip(self):
        raw = b"\x89PNG" + b"\x00" * 100
        b64 = base64.b64encode(raw).decode()
        req = PredictRequest(request_id="r", iris_b64=b64, fingerprint_b64=b64)
        assert req.iris_bytes() == raw

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):  # noqa: B017
            PredictRequest(request_id="r", iris_b64="not-valid-b64!!!", fingerprint_b64="abc")
