"""ObjectStore and BlobStore protocols — storage abstraction layer."""

from __future__ import annotations

from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    """Read/write raw blobs (Parquet shards, images) from a backing store.

    Implementations: AzureBlobIngestor (cloud), LocalParquetIngestor (local).
    """

    def upload(self, key: str, data: bytes | BinaryIO) -> None:
        """Write bytes under the given key."""
        ...

    def download(self, key: str) -> bytes:
        """Read bytes for the given key. Raises ShardNotFoundError if absent."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all keys matching the prefix."""
        ...

    def exists(self, key: str) -> bool:
        """Return True if the key exists in the store."""
        ...


@runtime_checkable
class ObjectStore(Protocol):
    """High-throughput object store used by Ray actors (Plasma / remote).

    In the local profile, this is backed by Ray's in-process Plasma store.
    In the cloud profile, this is backed by a multi-node Ray cluster.
    """

    def put(self, obj: object) -> str:
        """Serialise *obj* into the store and return an opaque reference string."""
        ...

    def get(self, ref: str) -> object:
        """Deserialise and return the object identified by *ref*."""
        ...
