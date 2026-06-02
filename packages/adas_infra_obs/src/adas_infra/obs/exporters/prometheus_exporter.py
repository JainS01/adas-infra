"""PrometheusExporter — HTTP exposition server for the shared CollectorRegistry.

Starts an HTTP server on *port* (default 9000) that serves the
/metrics endpoint in Prometheus text format.  The docker-compose
prometheus.yml scrapes this endpoint from the training container.
"""

from __future__ import annotations

import logging
import threading

from prometheus_client import start_http_server

from adas_infra.obs.metrics.registry import get_registry

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Wraps start_http_server for the shared registry on a background thread."""

    def __init__(self, port: int = 9000, addr: str = "0.0.0.0") -> None:
        self._port = port
        self._addr = addr
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        registry = get_registry()
        start_http_server(port=self._port, addr=self._addr, registry=registry)
        self._started = True
        logger.info("PrometheusExporter: serving metrics on %s:%d/metrics", self._addr, self._port)
