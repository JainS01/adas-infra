"""Contract layer: Protocols for collaborator contracts, ABCs for shared state."""

from adas_infra.core.contracts.cdc_source import CDCSource
from adas_infra.core.contracts.dataset import MultimodalDataset
from adas_infra.core.contracts.evaluator import Evaluator
from adas_infra.core.contracts.inference_endpoint import InferenceEndpoint
from adas_infra.core.contracts.ingestor import Ingestor
from adas_infra.core.contracts.manifest_store import BaseManifestStore
from adas_infra.core.contracts.model_registry import BaseModelRegistry
from adas_infra.core.contracts.object_store import BlobStore, ObjectStore
from adas_infra.core.contracts.trainer import BaseTrainer
from adas_infra.core.contracts.transform import Transform

__all__ = [
    "CDCSource",
    "MultimodalDataset",
    "Evaluator",
    "InferenceEndpoint",
    "Ingestor",
    "BaseManifestStore",
    "BaseModelRegistry",
    "BlobStore",
    "ObjectStore",
    "BaseTrainer",
    "Transform",
]
