"""Plasma throughput benchmark — pytest-benchmark regression gate."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def ray_env():
    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=2, object_store_memory=256 * 1024 * 1024, ignore_reinit_error=True)
    yield


def test_plasma_put_get_throughput(benchmark, ray_env):
    from adas_infra.bench.plasma_throughput import benchmark_plasma_put_get, assert_throughput_target

    results = benchmark(benchmark_plasma_put_get, num_batches=20, batch_size=16)
    assert_throughput_target(results, min_put_throughput=5.0)  # 5 batch/s minimum on any machine
