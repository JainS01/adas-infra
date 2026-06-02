"""CDCSource protocol — change-data-capture event stream."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from adas_infra.core.schemas.delta_record import DeltaRecord


@runtime_checkable
class CDCSource(Protocol):
    """Streams DeltaRecords as new data arrives in the backing store.

    Implementations: cdc_eventgrid.py (cloud), cdc_filesystem_watcher.py (local).
    """

    def stream(self) -> Iterator[DeltaRecord]:
        """Yield DeltaRecords; blocks until the next event arrives."""
        ...

    def stop(self) -> None:
        """Gracefully stop the CDC stream and release resources."""
        ...
