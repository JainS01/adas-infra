"""Shared test fixtures for all test suites."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory for synthetic data (avoids regenerating per test)."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def tmp_state_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("state")


@pytest.fixture(scope="session")
def tmp_ckpt_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("checkpoints")


@pytest.fixture(scope="session")
def synthetic_table(tmp_data_dir: Path):
    """Session-scoped synthetic Arrow table with 20 subjects, 5 samples each."""
    from adas_infra.data.ingestion.synthetic_ingestor import (
        SyntheticIngestor,
        SyntheticIngestorConfig,
    )

    cfg = SyntheticIngestorConfig(
        num_subjects=10,
        samples_per_subject=3,
        seed=42,
        output_dir=str(tmp_data_dir),
    )
    ingestor = SyntheticIngestor(cfg=cfg)
    shard_ids = ingestor.generate_shards(num_shards=2)
    return ingestor.ingest(shard_ids)


@pytest.fixture(scope="session")
def small_model():
    """Session-scoped tiny FusionBaseline for fast forward-pass tests."""
    from adas_infra.train.models.fusion_baseline import FusionBaseline

    return FusionBaseline(num_classes=10, iris_embed_dim=32, fp_embed_dim=32)


@pytest.fixture(scope="session")
def ray_session():
    """Initialize Ray once for the session; shut down at teardown."""
    import ray

    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            object_store_memory=256 * 1024 * 1024,
            ignore_reinit_error=True,
        )
    yield
    # Don't shutdown between tests — session fixture handles it
