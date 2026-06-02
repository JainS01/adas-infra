"""plasma_pressure — leading indicator of Plasma spill-to-disk events.

When this counter rises, the Plasma object store is evicting objects to disk,
which means the PlasmaPrefetcher is seeing high latency (see dataloader_lag_seconds).
The root cause is typically: batch_size too large, or not enough prefetcher slots.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter

_spill_counter: Counter | None = None


def register(registry: CollectorRegistry) -> None:
    global _spill_counter
    _spill_counter = Counter(
        "plasma_spill_events_total",
        "Number of times Ray Plasma spilled objects to disk (leading indicator of memory pressure)",
        registry=registry,
    )


def record_spill(count: int = 1) -> None:
    """Increment the spill counter. Called by the prefetcher on slow ray.get()."""
    if _spill_counter is not None:
        _spill_counter.inc(count)
