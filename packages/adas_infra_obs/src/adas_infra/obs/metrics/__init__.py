"""Prometheus metric modules. register_all() is called once at CLI startup."""

from adas_infra.obs.metrics.registry import get_registry, register_all

__all__ = ["get_registry", "register_all"]
