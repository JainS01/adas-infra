"""Data loaders.

Two interchangeable data paths behind the same batch-dict shape:
  - RayDatasetLoader + PlasmaPrefetcher — the high-throughput zero-copy default.
  - ArrowBiometricDataset + build_torch_dataloader — the standard-DataLoader
    fallback (single-device profile), typed against the MultimodalDataset contract.
"""

from adas_infra.data.loaders.sharded_sampler import ShardedSampler
from adas_infra.data.loaders.torch_dataset import (
    ArrowBiometricDataset,
    build_torch_dataloader,
)

__all__ = [
    "ArrowBiometricDataset",
    "ShardedSampler",
    "build_torch_dataloader",
]
