"""PlasmaPrefetcher — bounded async queue between Plasma and the training loop.

Specification (§0.6 of the architecture contract):
  - maxsize = max(2, 2 x micro_batches_per_step)
  - When full: producing Ray actor's put() is back-pressured (never drops)
  - Training step blocks on next() and records dataloader_lag_seconds
  - Plasma spill-to-disk is surfaced via plasma_pressure metric

The prefetcher runs entirely within the training worker process.  The
background thread calls ray.get(ref) which deserialises the batch from
Plasma shared memory.  When the actor and the training worker are co-located
(same node), this is a zero-copy operation — no data crosses a network link.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import ray

logger = logging.getLogger(__name__)

_SENTINEL = object()


class PlasmaPrefetcher:
    """Pre-fetches batches from Plasma ObjectRefs into a bounded in-process queue.

    Usage::

        prefetcher = PlasmaPrefetcher(object_refs, batch_size=32, micro_batches=4)
        for batch in prefetcher:
            loss = model(batch)   # batch is already in CPU RAM, no blocking ray.get

    The background thread saturates the queue before the training loop starts;
    the training loop never waits unless preprocessing falls behind (backpressure).
    """

    def __init__(
        self,
        object_refs: list[Any],
        micro_batches_per_step: int = 1,
        timeout_seconds: float = 30.0,
    ) -> None:
        maxsize = max(2, 2 * micro_batches_per_step)
        self._refs = object_refs
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._timeout = timeout_seconds
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._lag_callback: Any = None  # set by observability layer

    def register_lag_callback(self, callback: Any) -> None:
        """Register a callable(wait_seconds: float) for metric collection."""
        self._lag_callback = callback

    def start(self) -> None:
        """Spawn the background prefetch thread."""
        self._thread = threading.Thread(
            target=self._prefetch_worker,
            name="plasma-prefetcher",
            daemon=True,
        )
        self._thread.start()

    def __iter__(self) -> PlasmaPrefetcher:
        if self._thread is None:
            self.start()
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next batch, blocking until Plasma delivers it.

        Measures and reports the wait duration as dataloader_lag_seconds.
        """
        t0 = time.perf_counter()
        while True:
            try:
                item = self._queue.get(timeout=0.5)
                break
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError("PlasmaPrefetcher background error") from self._error
                if self._thread is not None and not self._thread.is_alive():
                    # Thread has finished; drain the queue
                    try:
                        item = self._queue.get_nowait()
                        break
                    except queue.Empty:
                        raise StopIteration from None
        if item is _SENTINEL:
            raise StopIteration
        lag = time.perf_counter() - t0
        if lag > 0.001 and self._lag_callback is not None:
            self._lag_callback(lag)
        return item  # type: ignore[no-any-return]

    def _prefetch_worker(self) -> None:
        """Background thread: resolves ObjectRefs from Plasma and enqueues batches."""
        try:
            for ref in self._refs:
                t0 = time.perf_counter()
                batch = ray.get(ref, timeout=self._timeout)
                elapsed = time.perf_counter() - t0
                if elapsed > 1.0:
                    logger.warning(
                        "PlasmaPrefetcher: slow ray.get — %.2fs (possible Plasma spill)",
                        elapsed,
                    )
                self._queue.put(batch)  # blocks when full — backpressure
        except Exception as exc:
            logger.error("PlasmaPrefetcher worker crashed: %s", exc)
            self._error = exc
        finally:
            self._queue.put(_SENTINEL)

    def __len__(self) -> int:
        return len(self._refs)
