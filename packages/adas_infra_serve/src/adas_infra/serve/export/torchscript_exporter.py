"""TorchScriptExporter — traces FusionBaseline to a portable TorchScript artifact.

TorchScript is preferred over ONNX for the local_mock path because:
  - Zero external runtime dependency (no onnxruntime install needed in CI)
  - Full Python fallback via torch.jit.load
  - Directly loadable by edge teams consuming the model registry artifact

The exporter validates the traced model by running a dummy inference pass
before writing the artifact, so broken exports are caught immediately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from adas_infra.core.errors import CheckpointError

logger = logging.getLogger(__name__)

_IRIS_DUMMY_SHAPE = (1, 1, 64, 64)
_FP_DUMMY_SHAPE = (1, 1, 96, 96)


class TorchScriptExporter:
    """Traces a model to TorchScript and writes it to *dest*."""

    def __init__(self, device: torch.device | None = None) -> None:
        self._device = device or torch.device("cpu")

    def export(
        self,
        model: nn.Module,
        dest: Path,
        validate: bool = True,
    ) -> Path:
        """Trace *model* and write the TorchScript archive to *dest*.

        Args:
            model:    Trained PyTorch model (moved to CPU before tracing)
            dest:     Output path for the .pt TorchScript file
            validate: If True, run a forward pass on the traced model to verify

        Returns:
            dest (the written path)
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = model.to("cpu").eval()

        dummy_iris = torch.zeros(_IRIS_DUMMY_SHAPE)
        dummy_fp = torch.zeros(_FP_DUMMY_SHAPE)

        try:
            with torch.no_grad():
                traced = torch.jit.trace(model_cpu, (dummy_iris, dummy_fp))
        except Exception as exc:
            raise CheckpointError(f"TorchScript trace failed: {exc}") from exc

        if validate:
            with torch.no_grad():
                out = traced(dummy_iris, dummy_fp)
            expected_dim = model_cpu.num_classes if hasattr(model_cpu, "num_classes") else -1
            if expected_dim > 0 and out.shape[-1] != expected_dim:
                raise CheckpointError(
                    f"Traced model output dim {out.shape[-1]} != expected {expected_dim}"
                )
            logger.info("TorchScriptExporter: validation passed — output shape %s", out.shape)

        traced.save(str(dest))
        logger.info("TorchScriptExporter: saved traced model to %s", dest)
        return dest

    @staticmethod
    def load(path: Path, device: torch.device | None = None) -> torch.jit.ScriptModule:
        """Load a TorchScript archive for inference."""
        dev = device or torch.device("cpu")
        model = torch.jit.load(str(path), map_location=dev)
        model.eval()
        return model
