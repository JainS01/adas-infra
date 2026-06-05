#!/usr/bin/env python3
"""seed_delta_log.py — populate the WAL and manifest store for development/CI.

Usage::

    # Judge / offline path (synthetic data, no Kaggle download needed)
    python scripts/seed_delta_log.py --synthetic --num-shards 8 --output-dir ./data/synthetic

    # Real Kaggle dataset (must be downloaded first)
    python scripts/seed_delta_log.py --dataset-root ./data/kaggle

The script writes:
  1. Parquet shards to --output-dir (synthetic mode)
  2. Delta records to ./state/delta_log.ndjson
  3. Manifest entries to ./state/manifest.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("seed_delta_log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed the delta log and manifest store")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data (no Kaggle needed)",
    )
    p.add_argument("--dataset-root", default=None, help="Path to real Kaggle dataset root")
    p.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of Parquet shards to generate (synthetic only)",
    )
    p.add_argument("--num-subjects", type=int, default=20, help="Number of synthetic subjects")
    p.add_argument("--samples-per-subject", type=int, default=5, help="Samples per subject")
    p.add_argument(
        "--output-dir",
        default="./data/synthetic",
        help="Output directory for synthetic data",
    )
    p.add_argument("--state-dir", default="./state", help="Directory for WAL and manifest DB")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    return p.parse_args()


def seed_synthetic(args: argparse.Namespace) -> list[str]:
    """Generate synthetic shards and return their IDs."""
    # Import here so the script can be run without uv sync from the repo root
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages/adas_infra_data/src"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages/adas_infra_core/src"))

    from adas_infra.data.ingestion.synthetic_ingestor import (
        SyntheticIngestor,
        SyntheticIngestorConfig,
    )

    cfg = SyntheticIngestorConfig(
        num_subjects=args.num_subjects,
        samples_per_subject=args.samples_per_subject,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    ingestor = SyntheticIngestor(cfg=cfg)
    shard_ids = ingestor.generate_shards(num_shards=args.num_shards)
    log.info("Generated %d shards in %s", len(shard_ids), args.output_dir)
    return shard_ids


def seed_kaggle(dataset_root: str) -> list[str]:
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages/adas_infra_data/src"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages/adas_infra_core/src"))
    from adas_infra.data.ingestion.iris_fingerprint_ingestor import IrisFingerprintIngestor

    ingestor = IrisFingerprintIngestor(dataset_root=dataset_root)
    shard_ids = ingestor.list_shards()
    log.info("Found %d subject shards in %s", len(shard_ids), dataset_root)
    return shard_ids


def write_delta_and_manifest(shard_ids: list[str], state_dir: Path) -> None:
    from adas_infra.core.schemas.delta_record import DeltaOperation, DeltaRecord
    from adas_infra.data.delta.delta_log import DeltaLog
    from adas_infra.data.delta.manifest_store_sqlite import ManifestStoreSQLite

    state_dir.mkdir(parents=True, exist_ok=True)
    wal = DeltaLog(wal_dir=state_dir)
    store = ManifestStoreSQLite(db_path=str(state_dir / "manifest.db"))

    for sid in shard_ids:
        rec = DeltaRecord(
            shard_id=sid,
            operation=DeltaOperation.ADD,
            path=sid,
            byte_size=0,
            num_rows=0,
        )
        wal.append(rec)
        store.record_delta(rec)

    log.info(
        "Seeded %d records — WAL offset=%d, manifest_db=%s",
        len(shard_ids),
        wal.offset,
        state_dir / "manifest.db",
    )
    store.close()


def main() -> None:
    args = parse_args()

    if args.synthetic:
        shard_ids = seed_synthetic(args)
    elif args.dataset_root:
        shard_ids = seed_kaggle(args.dataset_root)
    else:
        log.error("Must specify --synthetic or --dataset-root")
        sys.exit(1)

    write_delta_and_manifest(shard_ids, Path(args.state_dir))
    log.info("Done. Run `make judge-quickstart` to train and serve.")


if __name__ == "__main__":
    main()
