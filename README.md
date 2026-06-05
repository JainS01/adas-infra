# ADAS Multimodal AI Infrastructure

A production-shaped MLOps platform for a **multimodal biometric recognition** task
(iris + fingerprint). The point of this repository is **not** the model's accuracy —
it is the **infrastructure, data pipeline, and software design** around the model:
how data is ingested, validated, preprocessed in parallel, fed to the GPU without
stalls, trained reproducibly, served behind a stable contract, and observed.

> Dataset: [multimodal-iris-fingerprint-biometric-data](https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data).
> No download is required to evaluate this repo — a deterministic `SyntheticIngestor`
> produces schema-identical data so the full pipeline runs on CPU with no cloud and no Kaggle account.

---

## How this maps to the evaluation criteria

| Weight | Criterion | Where to look |
|---:|---|---|
| 25 | **System design & architecture** | Ports-and-adapters contracts in `adas_infra_core`, profile switching, [docs/architecture.md](docs/architecture.md), [docs/adr/](docs/adr/) |
| 20 | **Python engineering quality** | `mypy --strict` clean across 86 files, full ruff ruleset, Protocol/ABC contracts, no dead code |
| 15 | **Data loading & performance** | `RayDatasetLoader` + `PlasmaPrefetcher`, `dataloader_lag_seconds`, [Performance section](#5-data-loading--performance) |
| 15 | **ML workflow** | `BaseTrainer` template + `SingleDeviceEngine`, reproducibility 4-tuple, checkpoint/eval/hooks |
| 10 | **Multimodal handling** | `BiometricFrame` schema, two-tower `FusionBaseline`, typed `MissingModalityError` |
| 10 | **CI/CD & software quality** | 6 GitHub Actions jobs (lint, types, unit, integration, e2e, benchmarks), pre-commit, bandit, gitleaks |
| 5 | **Documentation & reasoning** | This file + ADRs + the [trade-offs / scaling](#9-bottlenecks-scalability--engineering-trade-offs) section |

Everything documented here corresponds to code that runs. The
["intentionally mocked"](#10-scope--honest-boundaries) section states plainly what is
real versus stubbed, in keeping with the brief's rule: *do not include anything that
cannot be explained.*

---

## 1. Judge Quickstart (no cloud, no GPU)

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/).
Docker is optional (only for the Grafana/Prometheus stack).

```bash
uv sync --all-packages          # install all 7 workspace packages
cp conf/.env.example .env.local # no real secrets needed for the local profile

make judge-quickstart           # seed → train → publish → serve → predict
```

`make judge-quickstart` runs, in order:

| Step | Command | Code path | What to look for |
|---|---|---|---|
| Seed | `scripts/seed_delta_log.py --synthetic --num-shards 8` | `SyntheticIngestor → DeltaLog → ManifestStoreSQLite` | `state/delta_log.ndjson`, `WAL offset=8` |
| Train | `adas-train trainer.max_steps=50` | `MergePlanner → RayDatasetLoader → PlasmaPrefetcher → SingleDeviceEngine` | `MergePlanner(incremental): N shards`, `data_load_fraction` |
| Publish | `adas-publish-model --run-id …` | `TorchScriptExporter → MLflowLocalRegistry` | `checkpoints/fusion.pt`, `mlruns/` |
| Serve | `adas-serve` | `LocalFastAPIEndpoint` | `200` from `/predict` |

Run the test suites the way CI does:

```bash
make test-unit          # 39 unit tests
make test-integration   # 7 Ray/Plasma + delta integration tests
make test-e2e           # full local pipeline (<180s on CPU)
make benchmark          # Plasma put/get throughput micro-benchmark
```

Optional observability stack:

```bash
make obs-up             # Prometheus :9090 + Grafana :3000 (admin/admin)
```

---

## 2. Repository layout — a `uv` workspace of 7 packages

Each package is independently versionable and depends only on contracts, never on
another package's concrete classes.

```
packages/
├── adas_infra_core    contracts (Protocol + ABC), Pydantic/PyArrow schemas, determinism, registry — no ML deps
├── adas_infra_data    ingestion, delta/WAL/CDC, Ray data plane, transforms, validation
├── adas_infra_train   engines, models, loops, hooks, checkpointing, reproducibility
├── adas_infra_serve   model registry, TorchScript export, FastAPI inference endpoint
├── adas_infra_obs     Prometheus metrics, Grafana dashboards
├── adas_infra_bench   hardware / Plasma throughput benchmarks
└── adas_infra_cli     thin Hydra-backed entrypoints (adas-train, adas-ingest, adas-serve, adas-publish-model)
```

`adas_infra_core` has **no torch/ray/cloud dependency** — it is pure contracts and
schemas, so every other package compiles and tests against it in isolation.

---

## 3. System design & architecture

### Ports and adapters

The dependency rule is one-directional: **concrete adapters depend on contracts in
`core`; nothing depends on a concrete adapter.** Contracts come in two flavours, chosen
per [ADR-0001](docs/adr/0001-contract-layer-protocol-vs-abc.md):

- **`Protocol`** for structural seams the framework only *calls* — e.g.
  `MultimodalDataset` (`__len__`/`__getitem__`), `ObjectStore`. Any object of the right
  shape satisfies them; no inheritance required.
- **`ABC` + template method** for lifecycles we *own* — e.g. `BaseTrainer.fit()` calls
  `_setup → _run_loop → _teardown`, and `BaseManifestStore` defines the transactional
  shard bookkeeping that SQLite (local) and Postgres (cloud) both implement.

### Profile switching with zero code forks

Wiring is data, not code. Hydra `_target_` strings select the concrete class per
profile; the engine, planner, and prefetcher never name a twin directly:

```bash
adas-train                       # profile=local_mock (default): SQLite, filesystem, CPU — implemented & tested
```

`conf/profile/cloud_prod.yaml` is a **design sketch** of the same seam for Azure
(Blob + Postgres + multi-node Ray). It shows *where* cloud adapters plug in, but it is
not implemented or composable today — several of its referenced config groups
(`manifest=postgres`, `serve=azureml_online`, `registry=azureml_registry`) are
intentionally absent, and there is only one concrete engine (`SingleDeviceEngine`); DDP/FSDP
are contract-anticipated extension points, not code. This is a conscious scope choice — see
[§10](#10-scope--honest-boundaries) — the value demonstrated is the *seam*, not a live Azure
deployment that can't be exercised in CI.

The cloud boundary is the **model registry**: training ends at
`BaseModelRegistry.register()`, and downstream consumers (cloud inference, edge teams)
bind only to that contract — they never see Ray, Plasma, or the delta log.

---

## 4. Multimodal data handling

A sample is one `BiometricFrame`, defined once as a Pydantic model **and** a matching
PyArrow schema (`BIOMETRIC_ARROW_SCHEMA`) so the same contract governs API validation and
the zero-copy columnar data plane:

```
schema_version, subject_id, sample_id, iris_bytes, fingerprint_bytes, label, split, source_shard
```

- **Modalities stay raw** (`iris_bytes`, `fingerprint_bytes`) through ingestion and are
  decoded inside preprocessing actors — keeping shards compact and decode parallel.
- **Missing modality is a typed error, not a silent `None`.** `IrisFingerprintIngestor`
  raises `MissingModalityError(subject, modality, path)` when either image is absent for a
  subject, so a bad pair is traceable rather than quietly dropped.
- **Late fusion, extensible to a third modality.** `FusionBaseline` is two independent
  encoders (`IrisEncoder`, `FingerprintEncoder`) whose embeddings are concatenated and
  classified. Adding face/voice later is "add an encoder, widen the concat" — no rewrite.
  `AdaptiveAvgPool` makes each encoder resolution-agnostic; the fusion head uses
  `LayerNorm` (batch-size-independent, identical in train/eval) rather than BatchNorm.

---

## 5. Data loading & performance

This is the heart of the assignment, and the design mirrors a real GPU-training data plane.

### The path

```
PyArrow shard(s)
   │  RayDatasetLoader.get_object_refs(split)
   ▼
ray.data.Dataset ── map_batches(decode → align) in Ray actor processes (separate from trainer)
   │  ray.put(batch)               ← preprocessed Arrow/NumPy batch into Plasma shared memory
   ▼
list[ObjectRef]
   │  PlasmaPrefetcher  (bounded queue, maxsize = max(2, 2 × micro_batches))
   ▼
TrainLoop.run(iterable)            ← ray.get(ref): cross-process, zero-copy when co-located
```

- **Zero-copy handoff.** Decode/augment run in Ray actor *processes*; the trainer only
  `ray.get(ref)`s from Plasma shared memory — no pickling, no memcpy when actor and trainer
  share a node (see [ADR-0002](docs/adr/0002-plasma-cross-process-requirement.md), which is
  why the local profile requires `num_cpus ≥ 2`).
- **Bounded backpressure, never silent drops.** `PlasmaPrefetcher` is a
  `queue.Queue(maxsize=…)`; when the GPU outruns preprocessing the producer blocks, and the
  wait is recorded as `dataloader_lag_seconds`. A full queue is the signal to scale
  preprocessing — not a dropped batch.

### Two interchangeable data paths, one contract

Both produce the identical batch-dict the training loop consumes, and both bind to the
`MultimodalDataset` Protocol — so the loop is agnostic to which is wired:

1. **`RayDatasetLoader` + `PlasmaPrefetcher`** — the high-throughput, multi-process default.
2. **`ArrowBiometricDataset` + `build_torch_dataloader`** — a standard map-style
   `DataLoader` fallback (single-device profile) using `ShardedSampler` for DDP-correct,
   deterministically-shuffled, rank-disjoint partitions.

### Built-in bottleneck analysis

Every run prints a timing breakdown and self-diagnoses:

```
TrainLoop: 50 steps | wall=12.3s | data_load=1.2s (10%) | compute=11.1s (90%) | throughput=4.1 steps/s
```

`data_load_fraction` is the interpretation key:

| Fraction | Meaning | Action |
|---|---|---|
| `< 30%` | compute-bound — GPU saturated | healthy |
| `> 30%` | data-bound — preprocessing too slow | raise `ray.num_cpus` or prefetch depth (the loop emits a `BOTTLENECK DETECTED` warning) |
| `> 60%` | likely Plasma spill-to-disk | raise `object_store_memory_mb` |

The same signal is exported as the `dataloader_lag_seconds` Prometheus histogram and
surfaced on the Grafana SLO dashboard.

### Parallel / distributed preprocessing

Preprocessing parallelism is `RayDatasetLoader(concurrency=…)` → a
`ray.data.TaskPoolStrategy(size=N)` of decode actors. Locally that's a 2-worker pool;
`conf/ray/cluster_autoscale.yaml` points the same code at a multi-node cluster. `PyArrow`
is the substrate end-to-end (columnar slices, `pc.*` filters) so preprocessing stays
vectorised and the Plasma payloads are zero-copy Arrow buffers.

---

## 6. Training pipeline & reproducibility

`BaseTrainer.fit()` is a template method; `SingleDeviceEngine` (CPU/single-GPU, the judge
path) is the **one concrete engine** and fills in `_setup`/`_run_loop`/`_teardown`. The ABC
anticipates DDP/FSDP engines, but those are extension points, not implemented classes.
`TrainLoop` is data-source-agnostic
(prefetcher, DataLoader, or a plain list in tests), supports AMP/grad-clip, and fires
`TrainingHook`s (logging, run-manifest, checkpoint) at step and epoch boundaries.

**Reproducibility 4-tuple** — every run is pinned to an exact (code, config, data,
environment) state, persisted both as MLflow tags and a `run_manifest.json` next to the
checkpoint:

```
(git_sha, hydra_config_hash, delta_log_offset, profile)
```

- `git_sha` / `hydra_config_hash` — captured at run end over the **fully-resolved** config
  (interpolations expanded, not `${…}` placeholders).
- `delta_log_offset` — stamped from the live manifest store's WAL position
  (`engine.set_data_provenance(manifest.get_wal_offset())`), so the tuple's data axis
  reflects exactly which deltas were in scope.
- `seed_everything()` seeds Python/NumPy/Torch (+CUDA), sets `PYTHONHASHSEED`, enables
  cuDNN-deterministic and `torch.use_deterministic_algorithms(warn_only=True)`;
  `worker_init_fn` decorrelates DataLoader workers.

**Delta-aware ingestion.** A WAL (`DeltaLog`, fsync'd NDJSON) + SQLite manifest track which
shards are pending. `MergePlanner` runs `incremental` by default — feeding only new shards
to the ingestor and preserving warm Plasma caches — with `data.merge_strategy=full_scan`
for cold-start reprocessing (see [ADR-0003](docs/adr/0003-delta-strategy.md)).

**Schema evolution.** Every schema carries `schema_version`; readers upcast through
`read_versioned()` (migration chain → version-bound check → validate), wired into
`DeltaLog.replay`, so a WAL written by an older build still parses after a bump
([ADR-0005](docs/adr/0005-schema-evolution.md)).

---

## 7. Serving

Training's only output to the outside world is a registered model. The serving layer binds
to that and nothing upstream:

- **`TorchScriptExporter`** — traces the model, validates the traced output shape, writes
  `fusion.pt`.
- **`MLflowLocalRegistry`** — SQLite-backed local twin of an Azure ML registry; the contract
  (`register()` → versioned URI) is identical to the cloud adapter.
- **`LocalFastAPIEndpoint`** — `POST /predict` (multimodal inference, top-k), `GET /health`,
  `GET /metrics` (Prometheus). Request/response use the same `PredictRequest`/`PredictResponse`
  Pydantic schemas as the Azure online endpoint; all server state lives in `app.state`
  (testable without `importlib.reload`).

---

## 8. MLOps foundations — CI, testing, metrics

Six GitHub Actions jobs gate `main`, each running the **same commands** the Makefile exposes
locally (no CI-only relaxations):

| Job | Gate |
|---|---|
| Lint & format | `ruff check` (full ruleset E,F,I,N,UP,ANN,S,B,A,C4,PT,RUF — no per-run ignores), `ruff format --check`, `bandit`, `gitleaks` (gating, allow-list in `.gitleaks.toml`) |
| MyPy | `mypy --strict` — 86 source files, clean |
| Unit | 39 tests (schemas, versioning, delta log, ingestor, dataset, model) |
| Integration | 7 tests (delta/merge + Ray/Plasma round-trip) |
| E2E | full local pipeline via `pipeline.py` (ingest → train → export → register) |
| Benchmarks | Plasma put/get throughput (`pytest-benchmark`) |

Test pyramid: `tests/{unit,integration,e2e,benchmarks}`, markers `gpu`/`slow`/`e2e`,
session-scoped Ray and synthetic-data fixtures. Metrics (`dataloader_lag_seconds`,
`gpu_efficiency`, `plasma_pressure`) live in `adas_infra_obs` and feed the committed Grafana
`training_slo.json` dashboard.

---

## 9. Bottlenecks, scalability & engineering trade-offs

A deliberate, honest read of where this design holds and where it would change at scale.

- **Eager Plasma materialisation is a judge-path choice, not the production pattern.**
  `RayDatasetLoader.get_object_refs()` materialises all batches into Plasma before returning
  — correct and simple at Kaggle scale. At petabyte scale the right pattern is a streaming
  generator that `ray.put()`s each batch only as the prefetcher queue has room, eliminating
  the upfront memory footprint. **That swap is one function behind the same
  `list[ObjectRef]` contract** and touches no other layer — the contract boundary is what
  makes the change cheap.
- **Plasma requires co-location.** Zero-copy holds only when actor and trainer share a node;
  cross-node, Plasma falls back to object transfer. The contract is unchanged, but the
  cluster topology (`conf/ray/*`) must keep preprocessing close to GPUs — a scheduling
  concern, surfaced via `plasma_pressure`.
- **Single-process Ray is unsupported by design** — the value is cross-process zero-copy, so
  the local profile mandates `num_cpus ≥ 2` rather than silently degrading.
- **Backpressure over buffering.** A bounded queue that blocks (and is *measured*) is
  preferable to an unbounded one that hides a slow pipeline until OOM. The cost is that a
  slow producer stalls the GPU — which is exactly the condition we want visible in
  `dataloader_lag_seconds`.
- **Closed-set classification head.** `FusionBaseline` classifies a fixed subject set; an
  open-set/verification setup (embedding + metric loss) is the real biometric framing but
  out of scope here, where the model is deliberately the least important part.

---

## 10. Scope & honest boundaries

Per the brief — nothing here is included that can't be explained. What is **real** vs.
**mocked**:

- **Real & exercised on CPU:** synthetic ingestion, delta/WAL/manifest, Ray/Plasma data
  plane, training/eval/checkpoint/hooks, reproducibility tuple, TorchScript export, local
  MLflow registry, FastAPI serving, Prometheus metrics, the full CI matrix.
- **A design sketch, not implemented:** `conf/profile/cloud_prod.yaml` and the Azure /
  Postgres / DDP-H100 story. It documents the intended seam and proves the *local* twin
  satisfies each contract, but the cloud adapters and several of their config groups are not
  written, and `profile=cloud_prod` will not compose as-is. It is included to show
  infrastructure-aware *design*, which the brief encourages — not as runnable cloud code.
- **Deliberately minimal:** model architecture and accuracy. The dataset's value is as a
  *multimodal* shape to engineer around; evaluation scores are explicitly not a goal.

See [docs/architecture.md](docs/architecture.md) for the component view and
[docs/adr/](docs/adr/) for the decision records behind the choices above.
