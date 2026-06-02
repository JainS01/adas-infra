# ADR-0003: Delta Control Plane Decoupled from Data Plane

**Status:** Accepted  
**Date:** 2025-06-02

## Context

In production ADAS pipelines, data arrives continuously (new drive sessions, camera uploads, annotation updates). The data loading pipeline must handle this gracefully without:
1. Re-reading shards that haven't changed
2. Invalidating in-memory Plasma caches unnecessarily
3. Requiring a full dataset rescan for each training run

## Decision

The **delta control plane** (WAL + manifest store + merge planner) is decoupled from the **data plane** (Ray Dataset, Plasma, PyTorch DataLoader).

```
Azure Event Grid / filesystem watcher
    ↓
DeltaRecord → DeltaLog (WAL, append-only)
    ↓
ManifestStore (SQLite / Postgres)
    ↓
MergePlanner.plan() → shard_id list
    ↓
Ingestor.ingest(shard_ids) → Arrow Table
    ↓
RayDatasetLoader → Plasma ObjectRefs
```

Key properties:
- The merge planner emits a **shard list**; the ingestor never queries Event Grid.
- `mark_ingested()` is called after successful Arrow table construction — not before.
- The WAL offset is captured in the `RunManifest` reproducibility 4-tuple.
- New shards append to the WAL without invalidating warm Plasma caches for existing shards.

## Consequences

- The local_mock profile swaps Event Grid for `watchdog` (250ms debounce + blake3 verify) and Postgres for SQLite. The merge planner and WAL code are **unchanged**.
- Schema evolution (ADR-0005) is the WAL's responsibility: every `DeltaRecord` carries `schema_version`.
- The WAL is a long-lived artifact; corruption detection via checksum replay is mandatory.
