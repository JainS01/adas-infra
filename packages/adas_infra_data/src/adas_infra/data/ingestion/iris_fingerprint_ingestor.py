"""IrisFingerprintIngestor — reads the Kaggle multimodal iris/fingerprint dataset.

Dataset layout (Kaggle: ninadmehendale/multimodal-iris-fingerprint-biometric-data):
    <root>/
        iris/
            <subject_id>/
                left/  <sample>.jpg
                right/ <sample>.jpg
        fingerprint/
            <subject_id>/
                <sample>.bmp | .png | .jpg

Every subject must have at least one iris image AND one fingerprint image.
A MissingModalityError is raised (not silently skipped) for any incomplete pair.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa

from adas_infra.core.errors import MissingModalityError, ShardNotFoundError
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA

logger = logging.getLogger(__name__)

_IMAGE_GLOB = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
_VAL_FRACTION = 0.15
_TEST_FRACTION = 0.05


class IrisFingerprintIngestor:
    """Reads the Kaggle iris+fingerprint dataset and returns Arrow tables.

    Shards correspond to subject-level directories; each shard_id is the
    subject folder name under the dataset root.
    """

    def __init__(self, dataset_root: str, **kwargs: Any) -> None:
        self._root = Path(dataset_root)
        self._iris_root = self._root / "iris"
        self._fp_root = self._root / "fingerprint"
        self._validate_layout()

    def _validate_layout(self) -> None:
        if not self._iris_root.exists():
            raise ShardNotFoundError("iris", str(self._iris_root))
        if not self._fp_root.exists():
            raise ShardNotFoundError("fingerprint", str(self._fp_root))

    def list_shards(self) -> list[str]:
        """Return subject_id strings (one per subject directory)."""
        iris_subjects = {p.name for p in self._iris_root.iterdir() if p.is_dir()}
        fp_subjects = {p.name for p in self._fp_root.iterdir() if p.is_dir()}
        return sorted(iris_subjects & fp_subjects)

    def ingest(self, shard_ids: list[str]) -> pa.Table:
        """Build an Arrow table for the requested subject shard IDs."""
        all_subjects = set(self.list_shards())
        rows: list[dict] = []  # type: ignore[type-arg]
        total = sum(1 for sid in shard_ids for _ in self._iter_subject(sid, all_subjects))

        n_test = max(1, int(total * _TEST_FRACTION))
        n_val = max(1, int(total * _VAL_FRACTION))
        row_idx = 0

        subject_label_map = {sid: i for i, sid in enumerate(sorted(all_subjects))}

        for sid in shard_ids:
            for iris_bytes, fp_bytes, sample_id in self._iter_subject(sid, all_subjects):
                if row_idx < n_test:
                    split = "test"
                elif row_idx < n_test + n_val:
                    split = "val"
                else:
                    split = "train"

                rows.append({
                    "schema_version": 1,
                    "subject_id": sid,
                    "sample_id": sample_id,
                    "iris_bytes": iris_bytes,
                    "fingerprint_bytes": fp_bytes,
                    "label": subject_label_map[sid],
                    "split": split,
                    "source_shard": sid,
                })
                row_idx += 1

        return self._rows_to_table(rows)

    def _iter_subject(
        self,
        subject_id: str,
        all_subjects: set[str],
    ):  # type: ignore[return]
        """Yield (iris_bytes, fp_bytes, sample_id) for every sample of one subject."""
        if subject_id not in all_subjects:
            raise ShardNotFoundError(subject_id, str(self._root))

        iris_paths = self._collect_images(self._iris_root / subject_id)
        fp_paths = self._collect_images(self._fp_root / subject_id)

        if not iris_paths:
            raise MissingModalityError(subject_id, "iris", str(self._iris_root / subject_id))
        if not fp_paths:
            raise MissingModalityError(
                subject_id, "fingerprint", str(self._fp_root / subject_id)
            )

        # Pair each iris image with the first fingerprint (simplest pairing strategy)
        fp_bytes = fp_paths[0].read_bytes()
        for iris_path in iris_paths:
            sample_id = f"{subject_id}_{iris_path.stem}"
            yield iris_path.read_bytes(), fp_bytes, sample_id

    @staticmethod
    def _collect_images(directory: Path) -> list[Path]:
        images: list[Path] = []
        if not directory.exists():
            return images
        for glob in _IMAGE_GLOB:
            images.extend(directory.rglob(glob))
        return sorted(images)

    @staticmethod
    def _rows_to_table(rows: list[dict]) -> pa.Table:  # type: ignore[type-arg]
        if not rows:
            return pa.table({f.name: [] for f in BIOMETRIC_ARROW_SCHEMA}, schema=BIOMETRIC_ARROW_SCHEMA)
        return pa.Table.from_pylist(rows, schema=BIOMETRIC_ARROW_SCHEMA)
