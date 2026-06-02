"""Single shared CollectorRegistry — all metrics live here (§0.8).

Rules (enforced by code review and tests):
  1. Each metric module exposes `register(registry: CollectorRegistry) -> None`.
  2. Metrics MUST NOT be created at import time (no module-level Counter/Histogram).
  3. CLI entrypoints call `register_all()` exactly once at startup.
  4. Tests that need metrics call `register_all(fresh=True)` to get a clean registry.
"""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry

logger = logging.getLogger(__name__)

_registry: CollectorRegistry | None = None


def get_registry() -> CollectorRegistry:
    """Return the shared registry; raises if register_all() has not been called."""
    if _registry is None:
        raise RuntimeError(
            "Metrics registry not initialised. Call register_all() at process startup."
        )
    return _registry


def register_all(fresh: bool = False) -> CollectorRegistry:
    """Create the shared registry and register all metric modules.

    Args:
        fresh: If True, discard any existing registry (use in tests only).
    """
    global _registry
    if _registry is not None and not fresh:
        return _registry

    _registry = CollectorRegistry(auto_describe=True)

    from adas_infra.obs.metrics import dataloader_lag, gpu_efficiency, plasma_pressure

    dataloader_lag.register(_registry)
    gpu_efficiency.register(_registry)
    plasma_pressure.register(_registry)

    logger.info("Observability: registered all metrics in CollectorRegistry")
    return _registry
