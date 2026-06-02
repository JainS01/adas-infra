# ADR-0001: Protocol vs ABC for the Contract Layer

**Status:** Accepted  
**Date:** 2025-06-02

## Context

The ADAS infrastructure has a two-tier contract layer:

1. **Collaborator contracts** — interfaces between loosely-coupled components
   (ingestor ↔ data loader, endpoint ↔ client).
2. **Template contracts** — shared lifecycle behaviour with mutable state
   (trainer, manifest store, model registry).

## Decision

Use **`typing.Protocol` (runtime_checkable)** for collaborator contracts.
Use **`abc.ABC`** only for contracts with shared state or template methods.

| Contract | Type | Reason |
|---|---|---|
| `Ingestor` | Protocol | Stateless; test doubles can be plain functions |
| `MultimodalDataset` | Protocol | Duck-typed; compatible with torch.utils.data.Dataset |
| `Transform` | Protocol | Composable stateless callables |
| `CDCSource` | Protocol | Event stream; mock with a generator |
| `BlobStore` / `ObjectStore` | Protocol | I/O adapters; easily replaced in tests |
| `InferenceEndpoint` | Protocol | Thin wrapper; TestClient covers this |
| `BaseTrainer` | ABC | Template method pattern; `fit()` orchestrates lifecycle |
| `BaseManifestStore` | ABC | Transactional state; abstract `record_delta` etc. |
| `BaseModelRegistry` | ABC | Versioned state; push/fetch semantics |

## Consequences

- Protocol types allow test doubles without inheritance, enabling faster unit tests.
- ABC types enforce the template method contract at instantiation time.
- Import discipline rule (§0.3): `contracts/` modules may only import from `schemas/`, `errors`, and stdlib — never from each other. This prevents the most common monorepo circular-import death spiral.
