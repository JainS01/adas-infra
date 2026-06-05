"""Standalone local-mock training pipeline — no Hydra, no GPU, no Kaggle data.

Designed for CI smoke tests: runs the full pipeline end-to-end on CPU using
synthetic biometric data.  All paths are created under a temp directory so
the working tree stays clean after the run.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# Disable Ray's automatic "uv run" worker propagation (Ray >=2.43). When the
# driver is launched via `uv run`, Ray otherwise re-launches each worker with
# `uv run` inside its runtime working dir, which re-resolves against the root
# pyproject.toml (dev-only, no ray) and crashes with ModuleNotFoundError. The
# deps are already installed in the active .venv, so workers should just reuse
# the current interpreter. Must be set before `import ray`.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

# Allow running directly from the src/ directory
_SRC = Path(__file__).parent
_PKGS = (
    "adas_infra_core",
    "adas_infra_data",
    "adas_infra_train",
    "adas_infra_serve",
    "adas_infra_obs",
)
for _pkg in _PKGS:
    _p = _SRC.parent / _pkg / "src"
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("pipeline")


def main() -> None:
    import ray

    with tempfile.TemporaryDirectory(prefix="adas_pipeline_") as tmp:
        base = Path(tmp)
        data_dir = base / "data"
        state_dir = base / "state"
        ckpt_dir = base / "checkpoints"
        mlruns_dir = base / "mlruns"

        for d in (data_dir, state_dir, ckpt_dir, mlruns_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Ray ────────────────────────────────────────────────────────────
        if not ray.is_initialized():
            # 256 MB Plasma store is a deliberate bound for the CI/local smoke run, not a
            # misconfiguration: it keeps memory predictable on a shared runner. Ray logs an
            # informational "object store is only N% of memory" notice here — expected and
            # harmless at toy scale; production sizing lives in conf/ray/*.yaml.
            ray.init(
                num_cpus=2,
                object_store_memory=256 * 1024 * 1024,
                ignore_reinit_error=True,
            )
        log.info("Ray initialised")

        # ── Synthetic data ─────────────────────────────────────────────────
        from adas_infra.data.ingestion.synthetic_ingestor import (
            SyntheticIngestor,
            SyntheticIngestorConfig,
        )

        ingestor_cfg = SyntheticIngestorConfig(
            num_subjects=10,
            samples_per_subject=4,
            seed=42,
            output_dir=str(data_dir),
        )
        ingestor = SyntheticIngestor(cfg=ingestor_cfg)
        shard_ids = ingestor.generate_shards(num_shards=4)
        log.info("Generated %d shards", len(shard_ids))

        # ── Delta log + manifest ───────────────────────────────────────────
        from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
        from adas_infra.data.delta.delta_log import DeltaLog
        from adas_infra.data.delta.manifest_store_sqlite import ManifestStoreSQLite
        from adas_infra.data.delta.merge_planner import MergePlanner, MergeStrategy

        manifest = ManifestStoreSQLite(db_path=str(state_dir / "manifest.db"))
        wal = DeltaLog(wal_dir=state_dir)

        for sid in shard_ids:
            rec = DeltaRecord(
                shard_id=sid,
                operation=DeltaOperation.ADD,
                path=sid,
                byte_size=0,
                num_rows=0,
            )
            wal.append(rec)
            manifest.record_delta(rec)

        # Incremental is the production default — only pending shards are ingested.
        # Fresh temp manifest ⇒ all shards pending ⇒ all selected on this first run.
        planner = MergePlanner(
            manifest_store=manifest,
            strategy=MergeStrategy.INCREMENTAL,
        )
        planned_shards = planner.plan() or shard_ids

        # ── Load + validate ────────────────────────────────────────────────
        from adas_infra.data.validation.schema_guard import SchemaGuard

        table = ingestor.ingest(planned_shards)
        SchemaGuard().validate(table, source="pipeline")
        planner.commit(planned_shards)
        log.info("Ingested %d rows", len(table))

        # ── Ray dataset loader + Plasma prefetcher ─────────────────────────
        from adas_infra.data.loaders.plasma_prefetcher import PlasmaPrefetcher
        from adas_infra.data.loaders.ray_dataset_loader import RayDatasetLoader
        from adas_infra.obs.metrics.dataloader_lag import make_callback

        loader = RayDatasetLoader(table=table, batch_size=16, concurrency=2)
        train_refs = loader.get_object_refs(split="train")
        val_refs = loader.get_object_refs(split="val")

        train_prefetcher = PlasmaPrefetcher(train_refs)
        train_prefetcher.register_lag_callback(make_callback())
        val_prefetcher = PlasmaPrefetcher(val_refs) if val_refs else None

        # ── Model ──────────────────────────────────────────────────────────
        import pyarrow.compute as pc

        from adas_infra.train.models.fusion_baseline import FusionBaseline

        num_classes = int(pc.max(table.column("label")).as_py()) + 1  # type: ignore[attr-defined]
        model = FusionBaseline(
            num_classes=num_classes,
            iris_embed_dim=64,
            fp_embed_dim=64,
        )

        # ── Train ──────────────────────────────────────────────────────────
        from adas_infra.train.engines.single_device_engine import SingleDeviceEngine

        # Alias so the class-body attribute name doesn't shadow the free
        # variable it reads from (otherwise `num_classes = num_classes` inside
        # the nested class raises NameError — class bodies don't capture an
        # enclosing local that the body also assigns).
        _num_classes = num_classes

        class _Cfg:
            class trainer:  # noqa: N801
                seed = 42
                max_steps = 10
                lr = 1e-3
                weight_decay = 1e-4
                checkpoint_dir = str(ckpt_dir)
                profile = "local_mock"

            class model:  # noqa: N801
                num_classes = _num_classes
                iris_embed_dim = 64
                fp_embed_dim = 64

        engine = SingleDeviceEngine(cfg=_Cfg())
        engine.set_data_provenance(delta_log_offset=manifest.get_wal_offset())
        manifest_result = engine.fit(
            model=model,
            train_data=train_prefetcher,
            val_data=val_prefetcher,
        )
        log.info("Training complete. Run ID: %s", manifest_result.run_id)

        # ── Export ─────────────────────────────────────────────────────────
        from adas_infra.serve.export.torchscript_exporter import TorchScriptExporter

        model_pt = ckpt_dir / "fusion.pt"
        TorchScriptExporter().export(engine.model or model, dest=model_pt)

        # ── Register ───────────────────────────────────────────────────────
        from adas_infra.serve.registry.mlflow_local_registry import MLflowLocalRegistry

        reg = MLflowLocalRegistry(tracking_uri=f"sqlite:///{mlruns_dir / 'mlflow.db'}")
        version_uri = reg.register(model_pt, manifest_result)
        log.info("Registered model: %s", version_uri)

        manifest.close()
        ray.shutdown()
        log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
