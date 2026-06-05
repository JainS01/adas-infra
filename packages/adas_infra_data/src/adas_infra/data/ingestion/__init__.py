"""Ingestor implementations."""

from adas_infra.data.ingestion.iris_fingerprint_ingestor import IrisFingerprintIngestor
from adas_infra.data.ingestion.local_parquet_ingestor import LocalParquetIngestor
from adas_infra.data.ingestion.synthetic_ingestor import SyntheticIngestor

__all__ = ["IrisFingerprintIngestor", "LocalParquetIngestor", "SyntheticIngestor"]
