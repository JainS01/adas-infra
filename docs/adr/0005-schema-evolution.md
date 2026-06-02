# ADR-0005: Schema Evolution via schema_version + Upcast Helpers

**Status:** Accepted  
**Date:** 2025-06-02

## Context

The ADAS pipeline stores data in Parquet shards, a NDJSON WAL, and SQLite/Postgres. Schemas will evolve over time (adding fields, renaming columns). Re-writing historical data is expensive and error-prone.

## Decision

Every Pydantic and PyArrow schema in `core/schemas/` carries a top-level `schema_version: int` field.

**Rules:**

1. Readers **MUST** accept any `schema_version <= CURRENT_VERSION`.
2. Readers **MUST** raise `SchemaVersionError` for `schema_version > CURRENT_VERSION`.
3. Migration callables in `_versioning.MIGRATION_REGISTRY` upcast older records to current.
4. Bumping `CURRENT_VERSION` requires adding a migration callable — not optional.

**Implementation:**

```python
@versioned_schema(current=2)
class BiometricFrame(BaseModel):
    schema_version: int = 2
    # ... new field added in v2 with a default ...
    split: str = "train"  # added in v2
```

Migration entry:
```python
MIGRATION_REGISTRY["BiometricFrame"] = {
    1: lambda d: {**d, "schema_version": 2, "split": "train"}
}
```

## Consequences

- WAL records written with `schema_version=1` remain valid indefinitely.
- Adding a new required field is a breaking change; it must have a default for migration.
- The `SchemaGuard` validates `schema_version` at ingestion time — breaking records are rejected before entering the pipeline.
