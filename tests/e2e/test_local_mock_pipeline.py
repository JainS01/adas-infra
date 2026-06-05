"""E2E pipeline test — full local_mock path in under 180s on CPU.

This test exercises the complete pipeline:
  SyntheticIngestor
    → DeltaLog + ManifestStoreSQLite + MergePlanner
    → SchemaGuard validation
    → Ray data plane (RayDatasetLoader + PlasmaPrefetcher)
    → FusionBaseline model
    → SingleDeviceEngine (10 steps on CPU)
    → TorchScriptExporter
    → MLflowLocalRegistry
    → LocalFastAPIEndpoint /health + /predict

All test-to-test state is passed through the shared `pipeline_state` fixture,
not via class-level attribute assignment.  This makes failures self-describing.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import ray
import torch
from PIL import Image

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180)]


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ray_init():
    """Ray must be initialised with num_cpus=2 for cross-process Plasma."""
    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            object_store_memory=256 * 1024 * 1024,
            ignore_reinit_error=True,
        )
    yield


@pytest.fixture(scope="module")
def pipeline_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("e2e_pipeline")
    return {
        "data": base / "data",
        "state": base / "state",
        "checkpoints": base / "checkpoints",
        "mlruns": base / "mlruns",
    }


@pytest.fixture(scope="module")
def pipeline_state() -> dict[str, Any]:
    """Mutable dict shared across all tests in the module (explicit dependency)."""
    return {}


def _make_png_b64(h: int, w: int) -> str:
    arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Pipeline stages ────────────────────────────────────────────────────────────


def test_01_synthetic_ingestor_generates_valid_table(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any]
) -> None:
    from adas_infra.data.ingestion.synthetic_ingestor import (
        SyntheticIngestor,
        SyntheticIngestorConfig,
    )
    from adas_infra.data.validation.schema_guard import SchemaGuard

    cfg = SyntheticIngestorConfig(
        num_subjects=10,
        samples_per_subject=4,
        seed=42,
        output_dir=str(pipeline_dirs["data"]),
    )
    ingestor = SyntheticIngestor(cfg=cfg)
    shard_ids = ingestor.generate_shards(num_shards=4)
    table = ingestor.ingest(shard_ids)

    SchemaGuard().validate(table, source="e2e_test")
    assert len(table) == 10 * 4
    assert "iris_bytes" in table.schema.names
    assert "fingerprint_bytes" in table.schema.names

    pipeline_state["table"] = table
    pipeline_state["shard_ids"] = shard_ids


def test_02_delta_log_and_manifest_store(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any]
) -> None:
    from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
    from adas_infra.data.delta.delta_log import DeltaLog
    from adas_infra.data.delta.manifest_store_sqlite import ManifestStoreSQLite
    from adas_infra.data.delta.merge_planner import MergePlanner, MergeStrategy

    shard_ids: list[str] = pipeline_state["shard_ids"]
    wal = DeltaLog(wal_dir=pipeline_dirs["state"])
    store = ManifestStoreSQLite(db_path=str(pipeline_dirs["state"] / "manifest.db"))

    for sid in shard_ids:
        rec = DeltaRecord(
            shard_id=sid,
            operation=DeltaOperation.ADD,
            path=sid,
            byte_size=0,
            num_rows=0,
        )
        wal.append(rec)
        store.record_delta(rec)

    planner = MergePlanner(manifest_store=store, strategy=MergeStrategy.INCREMENTAL)
    planned = planner.plan()
    assert len(planned) == len(shard_ids), (
        f"Expected {len(shard_ids)} pending shards, got {len(planned)}"
    )

    planner.commit(planned)
    pipeline_state["wal_offset"] = wal.offset
    store.close()


def test_03_ray_dataset_loader_produces_object_refs(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any], ray_init: None
) -> None:
    from adas_infra.data.loaders.ray_dataset_loader import RayDatasetLoader

    table = pipeline_state["table"]
    loader = RayDatasetLoader(table=table, batch_size=8, concurrency=2)
    train_refs = loader.get_object_refs(split="train")
    val_refs = loader.get_object_refs(split="val")

    assert len(train_refs) > 0, "No train ObjectRefs produced"

    batch = ray.get(train_refs[0])
    assert "iris" in batch
    assert "fingerprint" in batch
    assert "label" in batch
    assert batch["iris"].shape[1:] == (1, 64, 64), f"Unexpected iris shape: {batch['iris'].shape}"
    assert batch["fingerprint"].shape[1:] == (1, 96, 96)

    pipeline_state["train_refs"] = train_refs
    pipeline_state["val_refs"] = val_refs


def test_04_plasma_prefetcher_delivers_all_batches(
    pipeline_state: dict[str, Any], ray_init: None
) -> None:
    from adas_infra.data.loaders.plasma_prefetcher import PlasmaPrefetcher

    refs: list[Any] = pipeline_state["train_refs"]
    prefetcher = PlasmaPrefetcher(refs)
    batches = list(prefetcher)
    assert len(batches) == len(refs), f"Expected {len(refs)} batches, got {len(batches)}"
    for batch in batches:
        assert batch["iris"].dtype == np.float32
        assert batch["label"].dtype == np.int64


def test_05_single_device_engine_trains(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any], ray_init: None
) -> None:
    import pyarrow.compute as pc

    from adas_infra.data.loaders.plasma_prefetcher import PlasmaPrefetcher
    from adas_infra.train.engines.single_device_engine import SingleDeviceEngine

    table = pipeline_state["table"]
    num_classes = int(pc.max(table.column("label")).as_py()) + 1

    # SimpleNamespace, not nested classes: a class body cannot read an enclosing
    # function local (`num_classes = num_classes` raises NameError), and the engine
    # only needs attribute access, which SimpleNamespace provides.
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            seed=42,
            max_steps=10,
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            checkpoint_dir=str(pipeline_dirs["checkpoints"]),
            profile="local_mock",
        ),
        model=SimpleNamespace(
            num_classes=num_classes,
            iris_embed_dim=32,
            fp_embed_dim=32,
            name="fusion_baseline",
        ),
    )

    engine = SingleDeviceEngine(cfg=cfg)
    train_pf = PlasmaPrefetcher(pipeline_state["train_refs"])
    val_pf = PlasmaPrefetcher(pipeline_state["val_refs"]) if pipeline_state["val_refs"] else None

    manifest = engine.fit(model=None, train_data=train_pf, val_data=val_pf)

    assert manifest.run_id is not None
    assert manifest.git_sha is not None
    assert manifest.profile == "local_mock"

    pipeline_state["engine"] = engine
    pipeline_state["manifest"] = manifest


def test_06_torchscript_export_and_validation(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any]
) -> None:
    from adas_infra.serve.export.torchscript_exporter import TorchScriptExporter

    model = pipeline_state["engine"].model
    assert model is not None, "Engine has no trained model"

    export_path = pipeline_dirs["checkpoints"] / "fusion.pt"
    result_path = TorchScriptExporter().export(model, dest=export_path, validate=True)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
    pipeline_state["model_path"] = result_path


def test_07_mlflow_registry_registers_model(
    pipeline_dirs: dict[str, Path], pipeline_state: dict[str, Any]
) -> None:
    from adas_infra.serve.registry.mlflow_local_registry import MLflowLocalRegistry

    tracking_uri = f"sqlite:///{pipeline_dirs['mlruns'] / 'mlflow.db'}"
    reg = MLflowLocalRegistry(tracking_uri=tracking_uri)
    version_uri = reg.register(pipeline_state["model_path"], pipeline_state["manifest"])

    assert version_uri, "Registry returned an empty version URI"
    pipeline_state["version_uri"] = version_uri


def test_08_fastapi_endpoint_health_and_predict(
    pipeline_state: dict[str, Any],
) -> None:
    """Test /health and /predict by injecting state into app.state — no reload needed."""
    import os

    from fastapi.testclient import TestClient

    import adas_infra.serve.inference.local_fastapi_endpoint as ep
    from adas_infra.serve.inference.local_fastapi_endpoint import _AppState, _load_model

    model_path = str(pipeline_state["model_path"])
    os.environ["ADAS_MODEL_PATH"] = model_path

    ep.app.state.inference = _AppState(
        model=_load_model(model_path),
        model_version="e2e_test",
    )

    client = TestClient(ep.app, raise_server_exceptions=True)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    payload = {
        "schema_version": 1,
        "request_id": "e2e-test-001",
        "iris_b64": _make_png_b64(64, 64),
        "fingerprint_b64": _make_png_b64(96, 96),
        "top_k": 3,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["request_id"] == "e2e-test-001"
    assert len(body["predictions"]) == 3
    assert all(0.0 <= p["confidence"] <= 1.0 for p in body["predictions"])
    assert body["latency_ms"] >= 0.0


def test_09_run_manifest_json_written(
    pipeline_dirs: dict[str, Path],
) -> None:
    manifest_path = pipeline_dirs["checkpoints"] / "run_manifest.json"
    assert manifest_path.exists(), f"run_manifest.json not written to {manifest_path}"
    data = json.loads(manifest_path.read_text())
    assert data["profile"] == "local_mock"
    assert "git_sha" in data
    assert "hydra_config_hash" in data
    assert "delta_log_offset" in data


def test_10_timing_metrics_present_in_final_metrics() -> None:
    """TrainLoop must emit data_load_fraction for bottleneck analysis."""
    from adas_infra.train.loops.train_loop import TrainLoop
    from adas_infra.train.models.fusion_baseline import FusionBaseline

    model = FusionBaseline(num_classes=3, iris_embed_dim=16, fp_embed_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fake_batches = [
        {
            "iris": torch.randn(4, 1, 64, 64),
            "fingerprint": torch.randn(4, 1, 96, 96),
            "label": torch.randint(0, 3, (4,)),
        }
        for _ in range(3)
    ]
    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        max_steps=3,
    )
    metrics = loop.run(iter(fake_batches))

    assert "data_load_time_s" in metrics, "Timing split missing — bottleneck analysis impossible"
    assert "compute_time_s" in metrics
    assert "data_load_fraction" in metrics
    assert 0.0 <= metrics["data_load_fraction"] <= 1.0


def test_11_sample_json_matches_predict_schema() -> None:
    """tests/e2e/sample.json must be valid against PredictRequest."""
    from adas_infra.core.schemas.inference import PredictRequest

    sample_path = Path(__file__).parent / "sample.json"
    assert sample_path.exists(), "sample.json is missing from tests/e2e/"
    req = PredictRequest.model_validate(json.loads(sample_path.read_text()))
    assert req.request_id is not None
