"""PredictRequest / PredictResponse — single source of truth for the inference API.

This schema is shared by:
  - tests/e2e/sample.json (test fixture)
  - local_fastapi_endpoint.py (FastAPI handler)
  - azureml_online_endpoint.py (Azure ML scoring script)
  - docs/openapi.json (auto-generated)
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field, field_validator

CURRENT_VERSION = 1


class PredictRequest(BaseModel):
    """Multimodal biometric prediction request.

    Image bytes are base64-encoded to allow JSON serialisation.
    """

    schema_version: int = Field(default=CURRENT_VERSION)
    request_id: str = Field(description="Caller-supplied idempotency key")
    iris_b64: str = Field(description="Base64-encoded iris image bytes")
    fingerprint_b64: str = Field(description="Base64-encoded fingerprint image bytes")
    top_k: int = Field(default=1, ge=1, le=10, description="Return top-k predictions")

    @field_validator("iris_b64", "fingerprint_b64", mode="before")
    @classmethod
    def _validate_base64(cls, v: Any) -> str:
        if isinstance(v, bytes):
            return base64.b64encode(v).decode()
        try:
            base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("Field must be valid base64") from exc
        return str(v)

    def iris_bytes(self) -> bytes:
        return base64.b64decode(self.iris_b64)

    def fingerprint_bytes(self) -> bytes:
        return base64.b64decode(self.fingerprint_b64)


class PredictionCandidate(BaseModel):
    """One ranked prediction from the model."""

    rank: int = Field(ge=1)
    subject_id: str
    label: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Structured prediction response."""

    schema_version: int = Field(default=CURRENT_VERSION)
    request_id: str
    predictions: list[PredictionCandidate]
    model_version: str
    latency_ms: float = Field(ge=0.0)
