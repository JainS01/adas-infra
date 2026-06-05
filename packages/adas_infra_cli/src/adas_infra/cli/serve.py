"""adas-serve — start the local FastAPI inference endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model_path = OmegaConf.select(cfg, "serve.model_path", default=None)
    if model_path is None:
        last_path_file = Path(".last_model_path")
        if last_path_file.exists():
            model_path = last_path_file.read_text().strip()
        else:
            model_path = "./checkpoints/fusion.pt"

    port = int(OmegaConf.select(cfg, "serve.port", default=8080))
    log.info("Starting inference server: model=%s port=%d", model_path, port)

    from adas_infra.serve.inference.local_fastapi_endpoint import serve

    serve(model_path=str(model_path), port=port)
