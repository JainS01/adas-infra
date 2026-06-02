"""Prometheus metric modules. register_all() is called once at CLI startup."""

from adas_infra.obs.metrics.registry import register_all, get_registry

__all__ = ["register_all", "get_registry"]
