"""dataloader_lag_seconds — time the training loop waits for Plasma batches.

A rising lag indicates GPU starvation: the preprocessing pipeline cannot keep
pace with the training step.  The Grafana SLO dashboard alerts when p99 > 0.5s.
"""

from __future__ import annotations

from typing import Callable

from prometheus_client import CollectorRegistry, Histogram

_histogram: Histogram | None = None


def register(registry: CollectorRegistry) -> None:
    """Create the histogram and attach it to *registry*. Called once at startup."""
    global _histogram
    _histogram = Histogram(
        "dataloader_lag_seconds",
        "Time the training loop blocks waiting for the next Plasma batch",
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        registry=registry,
    )


def observe(lag_seconds: float) -> None:
    """Record one lag observation. No-op if register() has not been called."""
    if _histogram is not None:
        _histogram.observe(lag_seconds)


def make_callback() -> Callable[[float], None]:
    """Return a callable suitable for PlasmaPrefetcher.register_lag_callback()."""
    return observe
