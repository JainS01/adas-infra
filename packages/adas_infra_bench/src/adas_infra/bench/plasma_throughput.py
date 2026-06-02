"""Plasma throughput benchmark — measures Ray put()/get() throughput.

Run with: pytest tests/benchmarks/test_plasma_throughput.py --benchmark-json=results.json

This benchmark answers: "Can our preprocessing pipeline saturate a GPU?"
The target: > 500 batches/s put throughput for 32-sample batches on a single node.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import ray


def _make_dummy_batch(batch_size: int = 32) -> dict[str, Any]:
    return {
        "iris": np.random.rand(batch_size, 1, 64, 64).astype(np.float32),
        "fingerprint": np.random.rand(batch_size, 1, 96, 96).astype(np.float32),
        "label": np.random.randint(0, 20, size=(batch_size,), dtype=np.int64),
    }


def benchmark_plasma_put_get(
    num_batches: int = 100,
    batch_size: int = 32,
) -> dict[str, float]:
    """Measure put() and get() throughput in batches/second.

    Returns: {"put_throughput": float, "get_throughput": float, "roundtrip_ms": float}
    """
    if not ray.is_initialized():
        ray.init(num_cpus=2, object_store_memory=512 * 1024 * 1024, ignore_reinit_error=True)

    batches = [_make_dummy_batch(batch_size) for _ in range(num_batches)]

    # PUT benchmark
    t0 = time.perf_counter()
    refs = [ray.put(b) for b in batches]
    put_elapsed = time.perf_counter() - t0
    put_throughput = num_batches / put_elapsed

    # GET benchmark
    t0 = time.perf_counter()
    for ref in refs:
        _ = ray.get(ref)
    get_elapsed = time.perf_counter() - t0
    get_throughput = num_batches / get_elapsed

    roundtrip_ms = (put_elapsed + get_elapsed) / num_batches * 1000

    return {
        "put_throughput": put_throughput,
        "get_throughput": get_throughput,
        "roundtrip_ms": roundtrip_ms,
        "num_batches": float(num_batches),
        "batch_size": float(batch_size),
    }


def assert_throughput_target(
    results: dict[str, float],
    min_put_throughput: float = 50.0,
) -> None:
    """Assert that throughput meets the performance gate.

    The 50 batch/s lower bound is conservative for CPU-only CI.
    On H100 nodes the target is 500+ batch/s.
    """
    actual = results["put_throughput"]
    assert actual >= min_put_throughput, (
        f"Plasma put throughput {actual:.1f} batch/s < target {min_put_throughput} batch/s. "
        "Likely cause: object_store_memory too small or Plasma spilling to disk."
    )
