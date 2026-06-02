"""Ingestor implementations."""

from adas_infra.data.ingestion.synthetic_ingestor import SyntheticIngestor
from adas_infra.data.ingestion.iris_fingerprint_ingestor import IrisFingerprintIngestor
from adas_infra.data.ingestion.local_parquet_ingestor import LocalParquetIngestor

__all__ = ["SyntheticIngestor", "IrisFingerprintIngestor", "LocalParquetIngestor"]
