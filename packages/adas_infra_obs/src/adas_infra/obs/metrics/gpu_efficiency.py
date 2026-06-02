"""gpu_efficiency_percent — GPU utilisation via NVML; no-op on CPU runners.

Collected by a background thread every 5 seconds. The metric is zero on
CPU-only machines (CI, judge laptop), so the Grafana dashboard always renders
without missing data — it just shows 0%.
"""

from __future__ import annotations

import logging
import threading
import time

from prometheus_client import CollectorRegistry, Gauge

logger = logging.getLogger(__name__)

_gauge: Gauge | None = None
_thread: threading.Thread | None = None
_stop_event = threading.Event()


def register(registry: CollectorRegistry) -> None:
    global _gauge
    _gauge = Gauge(
        "gpu_efficiency_percent",
        "GPU utilisation percentage (NVML); 0 on CPU-only machines",
        labelnames=["device_index"],
        registry=registry,
    )
    _start_collector()


def _start_collector(interval_seconds: float = 5.0) -> None:
    global _thread
    _stop_event.clear()
    _thread = threading.Thread(
        target=_collect_loop,
        args=(interval_seconds,),
        name="gpu-efficiency-collector",
        daemon=True,
    )
    _thread.start()


def _collect_loop(interval: float) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        n_devices = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_devices)]

        while not _stop_event.is_set():
            for i, handle in enumerate(handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    if _gauge is not None:
                        _gauge.labels(device_index=str(i)).set(util.gpu)
                except pynvml.NVMLError:
                    pass
            time.sleep(interval)

        pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml not installed — gpu_efficiency_percent will stay 0")
        if _gauge is not None:
            _gauge.labels(device_index="0").set(0)
    except Exception as exc:
        logger.warning("GPU efficiency collector error: %s", exc)
        if _gauge is not None:
            _gauge.labels(device_index="0").set(0)
