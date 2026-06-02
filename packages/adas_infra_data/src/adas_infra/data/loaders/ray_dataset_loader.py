"""RayDatasetLoader — distributes preprocessing across Ray actors and stores
batches in Plasma (shared memory) for zero-copy handoff to the training worker.

Architecture (matches the component diagram, Panel 1):

    PyArrow table
        │
        ▼
    ray.data.Dataset   (lazy, partitioned by shard)
        │
        ▼  (map_batches — runs in Ray preprocessing actors, separate processes)
    decode → align → augment → oversample
        │
        ▼  (put into Plasma via ray.put)
    ObjectRefs  ──── pushed to PlasmaPrefetcher queue ────▶  Training worker

Zero-copy guarantee:  the Arrow→NumPy→tensor decode runs in Ray actor processes.
The training process only calls ray.get(ref) which de-serialises from shared
Plasma memory without a memcpy when actor and trainer share the same node.

Backpressure: actor put() calls block when Plasma is under pressure (Ray
enforces _max_pending_objects in actor scheduling); the PlasmaPrefetcher queue
signals wait via the dataloader_lag_seconds metric.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pyarrow as pa
import ray
import ray.data
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_NUM_CPUS_PER_ACTOR = 1
_IRIS_SIZE = (64, 64)
_FP_SIZE = (96, 96)


def _decode_image_safe(raw: bytes, size: tuple[int, int], modality: str, sample_id: str) -> np.ndarray:
    """Decode image bytes to a (1, H, W) float32 array; raises ValueError on corrupt/empty input."""
    if not raw:
        raise ValueError(f"Empty {modality} bytes for sample '{sample_id}'")
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize(size, Image.BILINEAR)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode {modality} for sample '{sample_id}': {exc}") from exc
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, :, :]  # (1, H, W)


def _decode_and_align_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Pure function applied inside Ray actor processes.

    Runs in a separate OS process from the trainer — this is where the
    zero-copy Plasma path materialises.

    Raises ValueError (surfaced as a Ray task error) on any missing or
    corrupt modality so the failure is traceable to the offending sample_id.
    """
    iris_list: list[np.ndarray] = []
    fp_list: list[np.ndarray] = []

    for iris_raw, fp_raw, sample_id in zip(
        batch["iris_bytes"], batch["fingerprint_bytes"], batch["sample_id"]
    ):
        iris_list.append(_decode_image_safe(bytes(iris_raw), _IRIS_SIZE, "iris", str(sample_id)))
        fp_list.append(_decode_image_safe(bytes(fp_raw), _FP_SIZE, "fingerprint", str(sample_id)))

    return {
        "iris": np.stack(iris_list, axis=0),            # (B, 1, 64, 64)
        "fingerprint": np.stack(fp_list, axis=0),        # (B, 1, 96, 96)
        "label": np.array(batch["label"], dtype=np.int64),
        "subject_id": batch["subject_id"],
        "sample_id": batch["sample_id"],
    }


class RayDatasetLoader:
    """Wraps a PyArrow table as a Ray Dataset pipeline.

    `get_object_refs(split)` returns a list of Plasma ObjectRefs, each pointing
    to a preprocessed batch dict already resident in shared memory.  The training
    worker calls `ray.get(ref)` to access batches without copying.

    Ray cluster config is injected via cfg; local profile uses `num_cpus=2`.
    """

    def __init__(
        self,
        table: pa.Table,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        num_cpus_per_actor: float = _DEFAULT_NUM_CPUS_PER_ACTOR,
        concurrency: int = 2,
    ) -> None:
        self._table = table
        self._batch_size = batch_size
        self._num_cpus_per_actor = num_cpus_per_actor
        self._concurrency = concurrency

    def get_object_refs(self, split: str = "train") -> list[Any]:
        """Build the Ray pipeline and materialise Plasma ObjectRefs for *split*.

        Returns a list of ray ObjectRefs; each ref holds one preprocessed batch dict.
        """
        import pyarrow.compute as pc

        mask = pc.equal(self._table.column("split"), split)
        split_table = self._table.filter(mask)

        if len(split_table) == 0:
            logger.warning("RayDatasetLoader: no rows for split=%s", split)
            return []

        ds: ray.data.Dataset = ray.data.from_arrow(split_table)

        ds = ds.map_batches(
            _decode_and_align_batch,
            batch_size=self._batch_size,
            batch_format="numpy",
            num_cpus=self._num_cpus_per_actor,
            concurrency=self._concurrency,
        )

        refs: list[Any] = []
        for batch in ds.iter_batches(batch_size=self._batch_size, batch_format="numpy"):
            ref = ray.put(batch)
            refs.append(ref)

        logger.info(
            "RayDatasetLoader: materialised %d Plasma ObjectRefs for split=%s",
            len(refs),
            split,
        )
        return refs
