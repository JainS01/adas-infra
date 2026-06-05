"""SyntheticIngestor — generates deterministic fake multimodal data for offline runs.

This ingestor is the backbone of the judge quickstart path. It produces
PyArrow tables that are structurally identical to the real Kaggle dataset,
allowing the entire pipeline (Ray, Plasma, training loop, FastAPI) to run
end-to-end without any external data dependency.
"""

from __future__ import annotations

import hashlib
import io
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from adas_infra.core.errors import ShardNotFoundError
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA

logger = logging.getLogger(__name__)

_IRIS_H, _IRIS_W = 64, 64
_FP_H, _FP_W = 96, 96
_VAL_FRACTION = 0.15
_TEST_FRACTION = 0.05


@dataclass
class SyntheticIngestorConfig:
    num_subjects: int = 20
    samples_per_subject: int = 5
    seed: int = 42
    output_dir: str = "./data/synthetic"


class SyntheticIngestor:
    """Generates and persists deterministic synthetic biometric data.

    Each shard is a Parquet file written under *output_dir*.  The ingestor
    is stateless after `generate_shards()` — calling `ingest(shard_ids)` simply
    reads back the pre-written files, which mirrors the real ingestor contract.

    Thread-safety: read-only after generation; safe to share across Ray workers.
    """

    def __init__(self, cfg: SyntheticIngestorConfig | None = None, **kwargs: Any) -> None:
        if cfg is None:
            valid_fields = {f.name for f in SyntheticIngestorConfig.__dataclass_fields__.values()}
            cfg = SyntheticIngestorConfig(**{k: v for k, v in kwargs.items() if k in valid_fields})
        self._cfg = cfg
        self._output_dir = Path(cfg.output_dir)
        self._shard_paths: dict[str, Path] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_shards(self, num_shards: int = 4) -> list[str]:
        """Generate *num_shards* Parquet files and return their shard IDs.

        Each shard receives a stable, deterministic ID derived from its index
        and the config seed so runs are reproducible.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        subjects_per_shard = max(1, self._cfg.num_subjects // num_shards)
        shard_ids: list[str] = []

        for shard_idx in range(num_shards):
            start = shard_idx * subjects_per_shard
            end = (
                self._cfg.num_subjects
                if shard_idx == num_shards - 1
                else start + subjects_per_shard
            )
            shard_id = self._shard_id(shard_idx)
            path = self._output_dir / f"{shard_id}.parquet"

            if not path.exists():
                table = self._build_table(range(start, end), shard_id)
                pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]
                logger.debug("Wrote shard %s → %s (%d rows)", shard_id, path, len(table))

            self._shard_paths[shard_id] = path
            shard_ids.append(shard_id)

        return shard_ids

    def ingest(self, shard_ids: list[str]) -> pa.Table:
        """Read the requested shards and concatenate into one Arrow table."""
        tables: list[pa.Table] = []
        for sid in shard_ids:
            path = self._resolve_shard(sid)
            tables.append(pq.read_table(path, schema=BIOMETRIC_ARROW_SCHEMA))  # type: ignore[no-untyped-call]
        if not tables:
            return pa.table({col.name: [] for col in BIOMETRIC_ARROW_SCHEMA})
        return pa.concat_tables(tables)

    def list_shards(self) -> list[str]:
        """Return the IDs of all shards currently written to disk."""
        existing: list[str] = []
        for p in sorted(self._output_dir.glob("shard_*.parquet")):
            existing.append(p.stem)
        return existing

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_table(self, subject_range: range, shard_id: str) -> pa.Table:
        subject_ids, sample_ids, iris_list, fp_list, labels, splits, shards = ([] for _ in range(7))
        total_samples = len(subject_range) * self._cfg.samples_per_subject
        n_val = max(1, int(total_samples * _VAL_FRACTION))
        n_test = max(1, int(total_samples * _TEST_FRACTION))
        row_idx = 0

        for subj_ord in subject_range:
            subj_str = f"S{subj_ord:04d}"
            for sample_idx in range(self._cfg.samples_per_subject):
                iris_bytes = self._random_image(_IRIS_H, _IRIS_W, seed=subj_ord * 1000 + sample_idx)
                fp_bytes = self._random_image(
                    _FP_H, _FP_W, seed=subj_ord * 1000 + sample_idx + 500_000
                )
                if row_idx < n_test:
                    split = "test"
                elif row_idx < n_test + n_val:
                    split = "val"
                else:
                    split = "train"

                subject_ids.append(subj_str)
                sample_ids.append(f"{subj_str}_s{sample_idx:02d}")
                iris_list.append(iris_bytes)
                fp_list.append(fp_bytes)
                labels.append(subj_ord)
                splits.append(split)
                shards.append(shard_id)
                row_idx += 1

        return pa.table(
            {
                "schema_version": pa.array([1] * row_idx, type=pa.int32()),
                "subject_id": pa.array(subject_ids, type=pa.string()),
                "sample_id": pa.array(sample_ids, type=pa.string()),
                "iris_bytes": pa.array(iris_list, type=pa.binary()),
                "fingerprint_bytes": pa.array(fp_list, type=pa.binary()),
                "label": pa.array(labels, type=pa.int64()),
                "split": pa.array(splits, type=pa.string()),
                "source_shard": pa.array(shards, type=pa.string()),
            },
            schema=BIOMETRIC_ARROW_SCHEMA,
        )

    @staticmethod
    def _random_image(h: int, w: int, seed: int) -> bytes:
        """Generate a grayscale PNG image with deterministic noise."""
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _shard_id(self, idx: int) -> str:
        payload = struct.pack(">II", self._cfg.seed, idx)
        digest = hashlib.sha256(payload).hexdigest()[:8]
        return f"shard_{idx:04d}_{digest}"

    def _resolve_shard(self, shard_id: str) -> Path:
        if shard_id in self._shard_paths:
            return self._shard_paths[shard_id]
        candidate = self._output_dir / f"{shard_id}.parquet"
        if candidate.exists():
            self._shard_paths[shard_id] = candidate
            return candidate
        raise ShardNotFoundError(shard_id=shard_id, location=str(self._output_dir))
