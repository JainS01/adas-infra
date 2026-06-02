"""LocalFastAPIEndpoint — production-parity REST inference server.

Serves the identical PredictRequest / PredictResponse schema as the Azure ML
online endpoint (§0.11).  The OpenAPI spec is auto-generated from these schemas
and committed at docs/openapi.json.

Endpoints:
  GET  /health           — liveness check
  POST /predict          — multimodal biometric inference
  GET  /metrics          — Prometheus text exposition (scraped by docker-compose)

The server loads the TorchScript model from the path specified in the
ADAS_MODEL_PATH environment variable, falling back to ./model/fusion.pt.

Model state is stored in app.state (not module globals) so the app is
testable without importlib.reload().
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError

from adas_infra.core.schemas.inference import (
    PredictRequest,
    PredictResponse,
    PredictionCandidate,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.getenv("ADAS_MODEL_PATH", "./model/fusion.pt")
_DEFAULT_PORT = int(os.getenv("ADAS_SERVE_PORT", "8080"))
_IRIS_SIZE = (64, 64)
_FP_SIZE = (96, 96)


@dataclass
class _AppState:
    """All mutable server state in one place — stored in app.state, not globals."""

    model: torch.jit.ScriptModule | None = None
    model_version: str = "unknown"
    subject_id_map: dict[int, str] = field(default_factory=dict)


def _load_model(model_path: str) -> torch.jit.ScriptModule:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Set ADAS_MODEL_PATH or run 'adas-train' first."
        )
    model = torch.jit.load(str(path), map_location=torch.device("cpu"))
    model.eval()
    logger.info("Loaded TorchScript model from %s", path)
    return model


def _decode_image(raw: bytes, size: tuple[int, int], label: str) -> torch.Tensor:
    """Decode image bytes to a (1, 1, H, W) float32 tensor; raises HTTPException on failure."""
    if not raw:
        raise HTTPException(status_code=422, detail=f"Empty {label} image bytes")
    try:
        img = Image.open(io.BytesIO(raw)).convert("L").resize(size, Image.BILINEAR)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode {label}: {exc}") from exc
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load model into app.state on startup; release on shutdown."""
    state = _AppState()
    model_path = os.getenv("ADAS_MODEL_PATH", _DEFAULT_MODEL_PATH)
    try:
        state.model = _load_model(model_path)
        state.model_version = Path(model_path).stem
        logger.info("Inference server ready: model_version=%s", state.model_version)
    except FileNotFoundError as exc:
        logger.warning("Model not found at startup — /predict will return 503: %s", exc)
    app.state.inference = state
    yield
    app.state.inference.model = None
    logger.info("Inference server shut down")


app = FastAPI(
    title="ADAS Biometric Inference API",
    version="0.1.0",
    description="Multimodal iris+fingerprint biometric identification",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness check — returns 200 when the model is loaded."""
    state: _AppState = request.app.state.inference
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": state.model_version}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """Score one multimodal biometric sample and return top-k predictions."""
    state: _AppState = request.app.state.inference
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    iris_tensor = _decode_image(body.iris_bytes(), _IRIS_SIZE, "iris")
    fp_tensor = _decode_image(body.fingerprint_bytes(), _FP_SIZE, "fingerprint")

    with torch.no_grad():
        logits = state.model(iris_tensor, fp_tensor)   # (1, num_classes)
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

    top_k = min(body.top_k, probs.shape[0])
    top_values, top_indices = torch.topk(probs, top_k)

    predictions = [
        PredictionCandidate(
            rank=rank + 1,
            subject_id=state.subject_id_map.get(idx.item(), f"S{idx.item():04d}"),
            label=idx.item(),
            confidence=round(val.item(), 6),
        )
        for rank, (val, idx) in enumerate(zip(top_values, top_indices))
    ]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return PredictResponse(
        request_id=body.request_id,
        predictions=predictions,
        model_version=state.model_version,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus metrics exposition (scraped by docker-compose Prometheus)."""
    try:
        from prometheus_client import generate_latest
        return generate_latest().decode("utf-8")
    except ImportError:
        return "# prometheus_client not installed\n"


def serve(
    model_path: str = _DEFAULT_MODEL_PATH,
    host: str = "0.0.0.0",
    port: int = _DEFAULT_PORT,
    subject_id_map: dict[int, str] | None = None,
) -> None:
    """Start the uvicorn server (blocking)."""
    os.environ["ADAS_MODEL_PATH"] = model_path
    if subject_id_map:
        # Inject before the lifespan context builds _AppState
        app.state.override_subject_id_map = subject_id_map
    uvicorn.run(app, host=host, port=port, log_level="info")
