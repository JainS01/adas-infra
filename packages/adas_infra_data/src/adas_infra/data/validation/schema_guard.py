"""SchemaGuard — validates Arrow tables against the expected schema.

Raises SchemaVersionError if a shard's schema_version exceeds CURRENT_VERSION,
and raises IngestionError for structural mismatches (missing columns, wrong types).
"""

from __future__ import annotations

import logging

import pyarrow as pa

from adas_infra.core.errors import IngestionError, SchemaVersionError
from adas_infra.core.schemas.frame import BIOMETRIC_ARROW_SCHEMA, CURRENT_VERSION

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {f.name for f in BIOMETRIC_ARROW_SCHEMA}


class SchemaGuard:
    """Validates an Arrow table for structural correctness and version compatibility."""

    def validate(self, table: pa.Table, source: str = "<unknown>") -> None:
        """Raise on any schema violation; return silently if the table is valid."""
        self._check_columns(table, source)
        self._check_version(table, source)
        self._check_non_empty_bytes(table, source)

    @staticmethod
    def _check_columns(table: pa.Table, source: str) -> None:
        actual = set(table.schema.names)
        missing = _REQUIRED_COLUMNS - actual
        if missing:
            raise IngestionError(
                f"Table from '{source}' is missing columns: {sorted(missing)}"
            )

    @staticmethod
    def _check_version(table: pa.Table, source: str) -> None:
        if "schema_version" not in table.schema.names:
            return
        import pyarrow.compute as pc
        max_ver = pc.max(table.column("schema_version")).as_py()
        if max_ver is not None and max_ver > CURRENT_VERSION:
            raise SchemaVersionError(
                found=int(max_ver),
                max_supported=CURRENT_VERSION,
                schema_name=f"BiometricFrame (from '{source}')",
            )

    @staticmethod
    def _check_non_empty_bytes(table: pa.Table, source: str) -> None:
        import pyarrow.compute as pc
        for col in ("iris_bytes", "fingerprint_bytes"):
            if col not in table.schema.names:
                continue
            null_count = pc.sum(pc.is_null(table.column(col))).as_py()
            if null_count and null_count > 0:
                raise IngestionError(
                    f"Table from '{source}': column '{col}' contains {null_count} null(s)"
                )
