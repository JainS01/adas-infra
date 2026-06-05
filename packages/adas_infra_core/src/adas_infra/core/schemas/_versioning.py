"""Schema evolution helpers — the @versioned_schema tag and the reader upcast path.

Wire-up (so nothing here is decorative):
  - ``@versioned_schema(current=N)`` stamps a Pydantic model with ``_CURRENT_VERSION``.
  - ``MIGRATION_REGISTRY`` maps each schema name to its {from_version: migrate_fn} chain.
  - ``read_versioned(Model, data)`` is what readers call: it upcasts a raw dict
    through the migration chain, rejects anything newer than the model supports,
    then validates. ``DeltaLog.replay`` uses it on every WAL record.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from adas_infra.core.errors import SchemaVersionError

T = TypeVar("T", bound=BaseModel)


def versioned_schema(current: int) -> Any:
    """Class decorator that stamps a Pydantic model with its current schema version.

    The stamp is read back by :func:`read_versioned` to bound-check incoming data.
    """

    def decorator(cls: type[T]) -> type[T]:
        cls._CURRENT_VERSION = current  # type: ignore[attr-defined]
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


def read_versioned(model_cls: type[T], data: dict[str, Any], schema_name: str | None = None) -> T:
    """Parse *data* into *model_cls*, upcasting through the migration chain first.

    The canonical reader entry point: handles forward-compatible data by running
    the registered migrations, rejects data newer than this build supports, then
    validates. Used by the WAL replay path so persisted records survive a schema bump.
    """
    name = schema_name or model_cls.__name__
    migrated = upcast_to_current(data, MIGRATION_REGISTRY.get(name, {}))
    current = int(getattr(model_cls, "_CURRENT_VERSION", 1))
    require_version_lte(int(migrated.get("schema_version", 1)), current, name)
    return model_cls.model_validate(migrated)


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
