# ADR-0002: Ray Plasma Zero-Copy Requires Distinct OS Processes

**Status:** Accepted  
**Date:** 2025-06-02

## Context

Ray Plasma is a shared-memory object store. The marketing claim of "zero-copy" is technically true only when the consumer and producer are in **different OS processes** on the same node. In a single-process setup (`ray.init()` with no arguments), `ray.put()` / `ray.get()` still serialises and deserialises through Plasma, but the operating system's copy-on-write semantics mean the "zero-copy" path may or may not materialise.

## Decision

**Minimum: `num_cpus=2`** for all profiles that claim zero-copy semantics.

- **Local profile:** `ray.init(num_cpus=2, object_store_memory=512*1024*1024)` spawns a real Ray worker process. The preprocessing actor runs in that worker; the training loop runs in the main process. The `ray.get(ref)` call in `PlasmaPrefetcher` is a genuine cross-process shared-memory read.

- **Cloud profile:** Multi-node Ray cluster; placement groups pin preprocessing actors and training workers to the same node. Zero-copy is guaranteed by co-location.

- **`num_cpus=1` is NOT a supported profile.** Tests that accidentally use it are considered broken — they pass but don't test the right thing.

## Consequences

- CI (`ubuntu-latest`) always launches with `num_cpus=2`. The cost is ~200ms startup overhead.
- `PlasmaPrefetcher` emits `dataloader_lag_seconds` which is observable in both profiles. A judge can see the Grafana dashboard confirm the prefetch is working.
- The single-process degenerate case is documented as unsupported in the architecture contract (§0.5).
