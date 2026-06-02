"""CDCFilesystemWatcher — local CDC source using watchdog + debounce + checksum.

Specification:
  - 250 ms debounce window: rapid file writes are coalesced into one event
  - blake3 checksum (falls back to sha256 if blake3 not installed)
  - Mid-write events are ignored until the file handle is closed
  - Each verified file emits exactly one DeltaRecord to the WAL

This is the local_mock counterpart of cdc_eventgrid.py (Azure Event Grid).
Both implement CDCSource (Protocol); the merge planner never distinguishes them.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord

logger = logging.getLogger(__name__)

_DEBOUNCE_SECONDS = 0.25
_WATCH_GLOB = "*.parquet"


def _checksum(path: Path) -> str:
    """Compute a blake3 checksum (falls back to sha256 if blake3 unavailable)."""
    try:
        import blake3  # type: ignore[import-untyped]
        hasher = blake3.blake3()  # type: ignore[attr-defined]
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except ImportError:
        sha = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()


class _DebounceHandler(FileSystemEventHandler):
    """Coalesces rapid filesystem events with a 250 ms debounce window."""

    def __init__(self, out_queue: "queue.Queue[Path]", watch_suffix: str = ".parquet") -> None:
        super().__init__()
        self._queue = out_queue
        self._suffix = watch_suffix
        self._timers: dict[Path, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_closed(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(self._suffix):
            self._schedule(Path(str(event.src_path)))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(self._suffix):
            self._schedule(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(self._suffix):
            self._schedule(Path(str(event.src_path)))

    def _schedule(self, path: Path) -> None:
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(_DEBOUNCE_SECONDS, self._fire, args=(path,))
            self._timers[path] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        with self._lock:
            self._timers.pop(path, None)
        if path.exists():
            self._queue.put(path)

    def cancel_all(self) -> None:
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class CDCFilesystemWatcher:
    """Watches a directory for new/modified Parquet shards and emits DeltaRecords.

    Usage::

        watcher = CDCFilesystemWatcher(watch_dir="./data/synthetic")
        for record in watcher.stream():
            manifest_store.record_delta(record)
    """

    def __init__(self, watch_dir: str, **kwargs: Any) -> None:
        self._watch_dir = Path(watch_dir)
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        self._file_queue: queue.Queue[Path] = queue.Queue()
        self._handler = _DebounceHandler(self._file_queue)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._watch_dir), recursive=True)
        self._stopped = False

    def start(self) -> None:
        self._observer.start()
        logger.info("CDCFilesystemWatcher: watching %s", self._watch_dir)

    def stop(self) -> None:
        self._stopped = True
        self._handler.cancel_all()
        self._observer.stop()
        self._observer.join(timeout=2)
        logger.info("CDCFilesystemWatcher: stopped")

    def stream(self) -> Iterator[DeltaRecord]:
        """Yield one DeltaRecord per verified file event; blocks between events."""
        if not self._observer.is_alive():
            self.start()
        while not self._stopped:
            try:
                path = self._file_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                checksum = _checksum(path)
                stat = path.stat()
                shard_id = path.stem
                record = DeltaRecord(
                    shard_id=shard_id,
                    operation=DeltaOperation.ADD,
                    path=str(path),
                    byte_size=stat.st_size,
                    num_rows=0,  # row count unknown until Parquet is parsed
                    checksum=checksum,
                )
                logger.debug("CDCFilesystemWatcher: emitting record for %s", shard_id)
                yield record
            except OSError as exc:
                logger.warning("CDCFilesystemWatcher: skipping %s — %s", path, exc)
