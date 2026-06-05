# Architecture

This document is the architectural reasoning behind the repo. The [README](../README.md)
covers *what* and *how to run*; this covers *why*. Decisions with lasting consequences are
recorded as [ADRs](adr/).

---

## 1. The governing rule: dependencies point at contracts

```
            ┌─────────────────────────── adas_infra_core ───────────────────────────┐
            │  contracts (Protocol + ABC) · schemas (Pydantic + PyArrow) · errors    │
            │  determinism · plugin registry            (NO torch / ray / cloud)     │
            └───────────────▲───────────────▲───────────────▲───────────────▲────────┘
                            │               │               │               │
                 adas_infra_data   adas_infra_train   adas_infra_serve   adas_infra_obs
                 (ingest, Ray,      (engines, loops,   (registry, export,  (metrics,
                  delta, Plasma)     reproducibility)   FastAPI)            dashboards)
                            ▲               ▲               ▲
                            └───────────────┴───────────────┴──── adas_infra_cli (Hydra wiring)
```

Every arrow points **inward, toward contracts**. No package imports another package's
concrete class. This is what makes the local↔cloud profile swap a configuration change
rather than a code change, and what lets each package be type-checked and tested in
isolation. `core` deliberately carries no ML or cloud dependency so the contract layer
can never be polluted by an implementation detail.

### Protocol vs. ABC — a deliberate split ([ADR-0001](adr/0001-contract-layer-protocol-vs-abc.md))

| Seam type | Mechanism | Examples | Rationale |
|---|---|---|---|
| Framework only **calls** it | `typing.Protocol` (structural) | `MultimodalDataset`, `ObjectStore`, `InferenceEndpoint` | duck-typing; an adapter needn't inherit anything, just match the shape |
| Framework **owns the lifecycle** | `abc.ABC` + template method | `BaseTrainer`, `BaseManifestStore` | shared orchestration lives in the base; subclasses fill in the holes |

`BaseTrainer.fit()` is the canonical template: it sequences
`_setup → _run_loop → _teardown` and owns shared state (config, hooks), while a concrete
engine provides only the device-specific bodies. **`SingleDeviceEngine` is the one engine
implemented**; the ABC names DDP/FSDP as anticipated subclasses, but they are extension
points, not code (`conf/trainer/ddp_h100.yaml` currently reuses `SingleDeviceEngine`).

---

## 2. The data plane (the performance-critical path)

```
 PyArrow shard(s)
     │
     ▼  RayDatasetLoader — ray.data.from_arrow → map_batches(decode→align)
 [ Ray actor processes ]                     compute = TaskPoolStrategy(size=N)   ← parallel preprocessing
     │
     ▼  ray.put(batch)                         ← Arrow/NumPy batch into Plasma shared memory
 list[ObjectRef]
     │
     ▼  PlasmaPrefetcher — Queue(maxsize = max(2, 2×micro_batches))
 [ trainer process ]   background thread: ray.get(ref)  ← zero-copy when co-located
     │                 records dataloader_lag_seconds; blocks producers when full (backpressure)
     ▼
 TrainLoop.run(iterable[batch_dict])
```

Three properties this layout buys, each mapped to a design decision:

1. **Cross-process zero-copy** ([ADR-0002](adr/0002-plasma-cross-process-requirement.md)) —
   decode runs off the trainer process; the trainer reads Plasma shared memory with no
   pickling/memcpy when co-located. Hence `num_cpus ≥ 2` is required, not optional.
2. **Bounded backpressure** — a full queue blocks producers and shows up as
   `dataloader_lag_seconds` rather than dropping batches or growing unbounded to OOM.
3. **Source-agnostic loop** — `TrainLoop` consumes any `iterable[batch_dict]`, so the
   Ray/Plasma path and the `ArrowBiometricDataset` + `build_torch_dataloader` fallback are
   interchangeable behind the `MultimodalDataset` contract.

### Self-diagnosing bottlenecks

`TrainLoop` separates data-wait from compute time and reports `data_load_fraction`. Below
30% is compute-bound (healthy); above 30% the loop emits `BOTTLENECK DETECTED` and the same
quantity is the `dataloader_lag_seconds` Prometheus histogram. The *infrastructure tells you
where the stall is* instead of leaving it to a profiler.

---

## 3. Delta-aware ingestion ([ADR-0003](adr/0003-delta-strategy.md))

```
new shard ──▶ DeltaLog.append (fsync'd NDJSON WAL) ──▶ ManifestStoreSQLite.record_delta (pending)
                                                              │
 MergePlanner.plan()  ── incremental (default): pending shards only ──▶ ingest ──▶ commit (mark ingested)
                       └─ full_scan (override):  every known shard  ──▶ reprocess / cold-start
```

The planner is a **pure function over manifest state** — it never touches storage, so it is
unit-testable without I/O and can run as a dry-run before binding an ingestor. `incremental`
is the runtime default in both `adas-train` and the smoke pipeline, which is what keeps warm
Plasma caches intact across runs; `full_scan` exists for reprocessing. The WAL offset feeds
the reproducibility tuple (next section). The local manifest is SQLite; the cloud twin is
Postgres — same `BaseManifestStore` ABC, no caller changes.

---

## 4. Reproducibility as a first-class output

Training does not just emit a checkpoint; it emits a **provenance record** good enough to
reconstruct the run:

```
RunManifest(git_sha, hydra_config_hash, delta_log_offset, profile, …)
   → MLflow tags   AND   run_manifest.json beside the checkpoint
```

- **Code:** `git_sha` (HEAD) — and the config is hashed *after* full Hydra resolution, so
  the hash reflects the values actually used, not unresolved `${…}` interpolations.
- **Data:** `delta_log_offset` is read from the live manifest store
  (`engine.set_data_provenance(...)`), tying the run to the exact WAL position — this is the
  axis most pipelines forget.
- **Environment:** `seed_everything()` covers Python/NumPy/Torch/CUDA + `PYTHONHASHSEED` +
  cuDNN determinism + `torch.use_deterministic_algorithms(warn_only=True)`.

Schema evolution is handled on the read side: `read_versioned()` runs the registered
migration chain, rejects anything newer than the build supports, then validates — wired into
`DeltaLog.replay` so persisted records survive a schema bump
([ADR-0005](adr/0005-schema-evolution.md)).

---

## 5. The cloud boundary is the model registry

```
[ data plane · Ray · Plasma · delta log ]  ──▶  BaseModelRegistry.register()  ──▶  [ cloud inference · edge teams ]
                  internal                            the contract                       downstream
```

Everything to the left is training-internal and may change freely. Downstream consumers bind
only to the registry contract — they never see Ray, Plasma, or the WAL. `local_mock` wires the
SQLite/filesystem/CPU twins and is the implemented, tested path; `cloud_prod` is a design
sketch of the Azure-twin wiring (see [README §10](../README.md#10-scope--honest-boundaries))
— the seam is real, the Azure adapters are not yet written. The engine, planner, and
prefetcher reference no concrete adapter either way.

---

## 6. Scaling posture (where this changes, and why it's cheap)

| Dimension | Local / judge | At scale | Cost of the change |
|---|---|---|---|
| Plasma fill | eager: materialise all refs up front | streaming generator: `put()` as queue drains | **one function**, same `list[ObjectRef]` contract |
| Preprocessing | 2-worker `TaskPoolStrategy` | multi-node autoscaling pool (`conf/ray/cluster_autoscale.yaml`) | config only |
| Manifest store | SQLite | Postgres Flexible Server | swap adapter behind `BaseManifestStore` |
| Storage | local NVMe parquet | Azure Blob | swap adapter behind the ingestor/object-store contract |
| Trainer | `SingleDeviceEngine` (CPU/1-GPU) | a DDP/FSDP engine (not yet written) | new `BaseTrainer` subclass; loop/hooks unchanged |

The recurring theme: **the contract boundary is where scale changes are absorbed.** That is
the property the architecture is optimised for, and the reason the model itself is kept
intentionally simple — it is the least interesting part of an ML *infrastructure* problem.
