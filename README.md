# ADAS Multimodal AI Infrastructure

Production-grade MLOps platform for L2+ ADAS multimodal biometric recognition.
Demonstrates scalable deep learning infrastructure design across seven independent packages.

---

## Judge Quickstart (no cloud, no GPU required)

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Docker (for the Grafana/Prometheus stack only — not required for training).

```bash
# 1. Clone and install all 7 packages in one shot (uv workspaces)
git clone <repo> && cd adas-infra
uv sync

# 2. Copy the env template (no real secrets needed for local profile)
cp conf/.env.example .env.local

# 3. (Optional) Start the Prometheus + Grafana observability stack
docker compose up -d

# 4. Run the complete pipeline end-to-end
make judge-quickstart
#   This runs four commands in order:
#     a. python scripts/seed_delta_log.py --synthetic --num-shards 8 --output-dir ./data/synthetic
#     b. adas-train +profile=local_mock trainer.max_steps=50
#     c. adas-publish-model --run-id $(cat .last_run_id)
#     d. curl localhost:8080/predict -d @tests/e2e/sample.json

# 5. Run the full e2e test suite (completes in <180s on CPU — CI green-badge path)
uv run pytest tests/e2e -q -m e2e

# 6. Open Grafana dashboards (if docker compose started)
open http://localhost:3000   # admin / admin
```

### What the quickstart does (step by step)

| Step | Code path | What to look for |
|---|---|---|
| Seed delta log | `SyntheticIngestor` → `DeltaLog` → `ManifestStoreSQLite` | `state/delta_log.ndjson` written |
| Train | `RayDatasetLoader` → `PlasmaPrefetcher` → `SingleDeviceEngine` | `data_load_fraction` in logs |
| Export | `TorchScriptExporter` | `checkpoints/fusion.pt` written |
| Register | `MLflowLocalRegistry` | `mlruns/` populated |
| Serve | `LocalFastAPIEndpoint` | JSON response from `/predict` |

---

## Architecture

Seven packages under one `uv` workspace, each independently versionable:

| Package | Responsibility |
|---|---|
| `adas_infra_core` | Contracts (Protocols + ABCs), Pydantic/PyArrow schemas, errors, determinism |
| `adas_infra_data` | Ingestion, delta/WAL/CDC watcher, Ray data plane, transforms, validation |
| `adas_infra_train` | Training engines (single-device / DDP / FSDP), hooks, checkpointing |
| `adas_infra_serve` | Model registry, TorchScript export, FastAPI inference endpoint |
| `adas_infra_obs` | Prometheus metrics, Grafana dashboards, OTEL tracing |
| `adas_infra_bench` | Hardware benchmarks (H100 vs H200 sweep), Plasma throughput |
| `adas_infra_cli` | Thin Hydra-backed CLI entrypoints |

## Profile switching (zero code changes)

```bash
# Local mock (default — judge path, no cloud, no GPU)
adas-train +profile=local_mock trainer.max_steps=50

# Cloud production (Azure Blob + Azure ML + multi-node Ray)
adas-train +profile=cloud_prod trainer=ddp_h100
```

Swapping `+profile=` is the only change. No code forks, no conditional imports.

---

## Performance Bottleneck Analysis

After every training run, the log contains a line like:

```
TrainLoop: 50 steps | wall=12.3s | data_load=1.2s (10%) | compute=11.1s (90%) | throughput=4.1 steps/s
```

**Interpreting `data_load_fraction`:**
- `< 30%` — compute-bound (normal, GPU is saturated)
- `> 30%` — data-bound (preprocessing pipeline is too slow; increase `ray.concurrency` or `PlasmaPrefetcher` queue depth)
- `> 60%` — likely Plasma spill-to-disk; increase `object_store_memory_mb` in `conf/ray/local_2worker.yaml`

The `dataloader_lag_seconds` Prometheus histogram and the Grafana SLO dashboard surface this in real time.

---

## Key Design Decisions

- **Zero-copy Ray/Plasma** — preprocessing actors `put()` Arrow batches into Plasma; training workers `get()` them cross-process. Both must be on the same node (`num_cpus ≥ 2`). Single-process Ray is not supported.
- **Bounded backpressure** — `PlasmaPrefetcher` is a `queue.Queue(maxsize=max(2, 2×micro_batches))`. When full it blocks producers — never drops silently.
- **Delta-aware ingestion** — WAL (`DeltaLog`) + SQLite manifest (`ManifestStoreSQLite`) track which shards are pending. `MergePlanner` only feeds new shards to the ingestor, preserving warm Plasma caches.
- **Reproducibility 4-tuple** — `(git_sha, hydra_config_hash, delta_log_offset, profile)` persisted as MLflow tags AND `run_manifest.json` next to the checkpoint.
- **Schema evolution** — every Pydantic/PyArrow schema carries `schema_version`; readers upcast through `_versioning.py`.
- **Missing modality = typed error** — `MissingModalityError` (not a silent None) is raised when either iris or fingerprint bytes are absent for a sample.
- **Plasma materialisation is eager by design (judge path)** — `RayDatasetLoader.get_object_refs()` materialises all batches into Plasma before returning, so the training loop starts with a fully pre-warmed queue. This is correct for the Kaggle dataset scale. At petabyte scale the right pattern is a streaming generator that lazily `put()`s each batch only as the `PlasmaPrefetcher` queue has room — eliminating the upfront memory footprint entirely. That change is a one-function swap behind the same `list[ObjectRef]` contract and does not affect any other layer.

See [docs/architecture.md](docs/architecture.md) and [docs/adr/](docs/adr/) for Architecture Decision Records.
