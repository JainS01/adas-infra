"""InferenceEndpoint protocol — unified prediction interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adas_infra.core.schemas.inference import PredictRequest, PredictResponse


@runtime_checkable
class InferenceEndpoint(Protocol):
    """Single-sample or batched prediction endpoint.

    Implementations: LocalFastAPIEndpoint, AzureMLOnlineEndpoint.
    """

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Score one request synchronously and return a structured response."""
        ...

    def health(self) -> bool:
        """Return True if the endpoint is ready to serve predictions."""
        ...
