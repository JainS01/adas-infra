"""Transform protocol — stateless or stateful data transformation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pyarrow as pa


@runtime_checkable
class Transform(Protocol):
    """A single composable transform applied to a PyArrow RecordBatch or sample dict."""

    def __call__(self, batch: pa.RecordBatch | dict[str, Any]) -> pa.RecordBatch | dict[str, Any]:
        """Apply the transform and return the same type as the input."""
        ...
