# Architecture Overview

## Seven-Package Monorepo

```
adas-infra/
├── packages/
│   ├── adas_infra_core     — contracts, schemas, determinism (no ML deps)
│   ├── adas_infra_data     — ingestion, Ray data plane, delta/WAL
│   ├── adas_infra_train    — engines, models, loops, hooks
│   ├── adas_infra_serve    — registry, export, FastAPI endpoint
│   ├── adas_infra_obs      — Prometheus, Grafana
│   ├── adas_infra_bench    — hardware benchmarks
│   └── adas_infra_cli      — Hydra-backed entrypoints
```

## Critical Design Decisions

### 1. Zero-Copy Plasma Path

Preprocessing runs in Ray actor processes (separate from the trainer). Arrow batches are `ray.put()` into Plasma shared memory. The training loop calls `ray.get(ref)` — a cross-process shared-memory read with no memcpy.

`PlasmaPrefetcher` maintains a bounded async queue in front of the training loop. This queue:
- Blocks producers when full (backpressure — never drops)
- Records `dataloader_lag_seconds` (visible in Grafana)

### 2. Bounded Backpressure

```
preprocessing actors  →  Plasma  →  PlasmaPrefetcher queue (maxsize=2×micro_batches)  →  TrainLoop
```

When the GPU is fast and preprocessing is slow, the queue fills up and actors block. `dataloader_lag_seconds` rises — this is the signal to scale up preprocessing.

### 3. Reproducibility 4-Tuple

```
(git_sha, hydra_config_hash, delta_log_offset, profile)
```

Persisted as MLflow tags AND `run_manifest.json` next to the checkpoint. Makes any training run attributable to a precise (code, config, data, environment) state.

### 4. Profile Switching Without Code Forks

`+profile=local_mock` wires every contract to its local twin (SQLite, filesystem, CPU). `+profile=cloud_prod` wires the Azure twin (Postgres, Azure Blob, H100). The training engine, merge planner, and prefetcher never reference either concrete class.

### 5. The Cloud Boundary is the Model Registry

Training ends at `BaseModelRegistry.register()`. Downstream consumers (cloud inference, edge teams) bind only to the registry contract. They never see Ray, Plasma, or the delta log.
