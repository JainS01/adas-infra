"""Evaluator protocol — post-training regression gate."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class Evaluator(Protocol):
    """Computes metrics over a holdout set and decides whether to promote a model."""

    def evaluate(self, model_uri: str, holdout: pa.Table) -> dict[str, float]:
        """Return a dict of metric_name → value (higher-is-better convention)."""
        ...

    def passes_gate(self, metrics: dict[str, float]) -> bool:
        """Return True iff the metrics clear all regression thresholds."""
        ...
