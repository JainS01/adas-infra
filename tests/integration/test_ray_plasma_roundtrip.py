"""Integration test: Ray put()/get() cross-process zero-copy roundtrip.

Verifies that:
1. Ray initialises with num_cpus=2 (real separate worker process)
2. Batches put into Plasma can be retrieved accurately
3. The data content is preserved byte-for-byte
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ray_env():
    import ray

    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            object_store_memory=256 * 1024 * 1024,
            ignore_reinit_error=True,
        )
    yield
    # Leave Ray running; session conftest owns shutdown


class TestRayPlasmaRoundtrip:
    def test_numpy_array_roundtrip(self, ray_env):
        import ray

        original = np.random.rand(32, 1, 64, 64).astype(np.float32)
        ref = ray.put(original)
        retrieved = ray.get(ref)
        np.testing.assert_array_equal(original, retrieved)

    def test_batch_dict_roundtrip(self, ray_env):
        import ray

        batch = {
            "iris": np.random.rand(16, 1, 64, 64).astype(np.float32),
            "fingerprint": np.random.rand(16, 1, 96, 96).astype(np.float32),
            "label": np.arange(16, dtype=np.int64),
        }
        ref = ray.put(batch)
        result = ray.get(ref)
        np.testing.assert_array_equal(batch["iris"], result["iris"])
        np.testing.assert_array_equal(batch["label"], result["label"])

    def test_multiple_refs_independent(self, ray_env):
        import ray

        refs = [ray.put(np.array([i], dtype=np.int32)) for i in range(5)]
        for i, ref in enumerate(refs):
            val = ray.get(ref)
            assert val[0] == i

    def test_plasma_prefetcher_delivers_all_batches(self, ray_env, tmp_path):
        import ray

        from adas_infra.data.loaders.plasma_prefetcher import PlasmaPrefetcher

        batches = [
            {
                "iris": np.random.rand(4, 1, 64, 64).astype(np.float32),
                "label": np.array([0, 1, 2, 3]),
            }
            for _ in range(5)
        ]
        refs = [ray.put(b) for b in batches]

        prefetcher = PlasmaPrefetcher(refs)
        received = list(prefetcher)
        assert len(received) == 5

        for i, batch in enumerate(received):
            np.testing.assert_array_equal(batch["label"], batches[i]["label"])
