"""Schema evolution helpers — upcast functions and the @versioned_schema decorator."""

from __future__ import annotations

import functools
from typing import Any, TypeVar

from adas_infra.core.errors import SchemaVersionError

T = TypeVar("T")


def versioned_schema(current: int) -> Any:
    """Class decorator that attaches version-guard helpers to a Pydantic model."""

    def decorator(cls: type[T]) -> type[T]:
        cls._CURRENT_VERSION = current  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def from_dict_versioned(klass: type[T], data: dict[str, Any]) -> T:
            """Parse *data*, rejecting any schema_version > current."""
            found = int(data.get("schema_version", 1))
            if found > current:
                raise SchemaVersionError(
                    found=found,
                    max_supported=current,
                    schema_name=klass.__name__,
                )
            return klass.model_validate(data)  # type: ignore[attr-defined]

        cls.from_dict_versioned = from_dict_versioned  # type: ignore[attr-defined]
        return cls

    return decorator


def upcast_to_current(data: dict[str, Any], migrations: dict[int, Any]) -> dict[str, Any]:
    """Sequentially apply migration callables to advance *data* to the current version.

    *migrations* maps {from_version: callable(data) -> data}.
    Example: {1: add_split_column, 2: rename_checksum_field}
    """
    version = int(data.get("schema_version", 1))
    for from_ver, migrate_fn in sorted(migrations.items()):
        if version == from_ver:
            data = migrate_fn(data)
            version = int(data.get("schema_version", version + 1))
    return data


def require_version_lte(found: int, maximum: int, schema_name: str) -> None:
    """Raise SchemaVersionError if *found* exceeds *maximum*."""
    if found > maximum:
        raise SchemaVersionError(found=found, max_supported=maximum, schema_name=schema_name)


# ── Concrete migration stubs (extend as schemas evolve) ──────────────────────


def _noop_migration(data: dict[str, Any]) -> dict[str, Any]:
    return data


# Registry of per-schema migration chains keyed by schema class name.
# When BiometricFrame bumps to v2, add: "BiometricFrame": {1: _add_v2_field}
MIGRATION_REGISTRY: dict[str, dict[int, Any]] = {
    "BiometricFrame": {},
    "DeltaRecord": {},
    "RunManifest": {},
    "PredictRequest": {},
}
