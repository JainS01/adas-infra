"""ColumnarWriter — appends RecordBatches to a Parquet file with schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA

logger = logging.getLogger(__name__)


class ColumnarWriter:
    """Streaming Parquet writer for BiometricFrame batches.

    Uses row-group buffering so the resulting file is efficiently splittable
    by downstream Arrow readers. Safe for single-threaded use only.
    """

    def __init__(
        self,
        dest: Path,
        schema: pa.Schema = BIOMETRIC_ARROW_SCHEMA,
        row_group_size: int = 1024,
        compression: str = "snappy",
    ) -> None:
        self._dest = dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(dest), schema, compression=compression
        )
        self._row_group_size = row_group_size
        self._rows_written = 0
        self._closed = False

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if self._closed:
            raise RuntimeError("ColumnarWriter is already closed")
        table = pa.Table.from_batches([batch], schema=self._writer.schema_arrow)
        self._writer.write_table(table)
        self._rows_written += len(batch)
        logger.debug("ColumnarWriter: wrote %d rows, total %d", len(batch), self._rows_written)

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True
            logger.info("ColumnarWriter: closed %s (%d rows total)", self._dest, self._rows_written)

    def __enter__(self) -> "ColumnarWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
