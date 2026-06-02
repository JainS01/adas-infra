"""adas-ingest — seed the delta log and manifest store from a data source."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../../../../../../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from adas_infra.data.ingestion.synthetic_ingestor import SyntheticIngestor, SyntheticIngestorConfig
    from adas_infra.data.delta.delta_log import DeltaLog
    from adas_infra.data.delta.manifest_store_sqlite import ManifestStoreSQLite
    from adas_infra.core.schemas.delta_record import DeltaRecord, DeltaOperation

    data_dir = Path(OmegaConf.select(cfg, "data.root", default="./data/synthetic"))
    state_dir = Path(OmegaConf.select(cfg, "storage.state_dir", default="./state"))
    num_shards = int(OmegaConf.select(cfg, "data.num_shards", default=4))

    ingestor_cfg = SyntheticIngestorConfig(
        num_subjects=int(OmegaConf.select(cfg, "data.num_subjects", default=20)),
        samples_per_subject=int(OmegaConf.select(cfg, "data.samples_per_subject", default=5)),
        seed=int(OmegaConf.select(cfg, "data.seed", default=42)),
        output_dir=str(data_dir),
    )
    ingestor = SyntheticIngestor(cfg=ingestor_cfg)
    shard_ids = ingestor.generate_shards(num_shards=num_shards)

    wal = DeltaLog(wal_dir=state_dir)
    manifest = ManifestStoreSQLite(db_path=str(state_dir / "manifest.db"))

    for sid in shard_ids:
        rec = DeltaRecord(shard_id=sid, operation=DeltaOperation.ADD, path=sid, byte_size=0, num_rows=0)
        wal.append(rec)
        manifest.record_delta(rec)

    manifest.close()
    log.info("Ingest complete: %d shards, WAL offset=%d", len(shard_ids), wal.offset)
