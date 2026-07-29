"""
Embedding storage for the BioNexus Embeddings module.

This module persists :class:`~embeddings.models.EmbeddingRecord` instances
and provides retrieval primitives (by ID, by source document, and "list
all") that ``embedding_search.py`` builds on top of to perform semantic
similarity search.

Storage backend
----------------
The concrete implementation provided here, :class:`JSONEmbeddingStore`,
persists records to a single JSON file on disk. This is intentionally
simple and dependency-free (no database server, no FAISS installation
required yet) so Phase 3 can be developed and tested immediately.

The store is designed against an abstract interface
(:class:`BaseEmbeddingStore`) specifically so that a FAISS-backed (or
other vector-database-backed) implementation can be introduced later
as a drop-in replacement, without requiring any changes to
``embedding_search.py`` or to frontend code that depends on
``BaseEmbeddingStore``. This satisfies the Phase 3 goal of "preparing
the system for a FAISS vector database" without prematurely introducing
that dependency now.

Classes
-------
BaseEmbeddingStore
    Abstract interface for embedding persistence and retrieval.
JSONEmbeddingStore
    Concrete JSON-file-backed implementation.

Exceptions
----------
EmbeddingStoreError
    Raised for storage-layer failures (I/O errors, corrupt data, etc.).
EmbeddingNotFoundError
    Raised when a lookup by ID finds no matching record.
DuplicateEmbeddingError
    Raised when attempting to add a record whose ID already exists.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from embeddings.models import EmbeddingMetadata, EmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingStoreError(RuntimeError):
    """Raised when the embedding store fails to read or write data."""


class EmbeddingNotFoundError(KeyError):
    """Raised when no embedding record exists for a given ``embedding_id``."""


class DuplicateEmbeddingError(ValueError):
    """Raised when adding a record whose ``embedding_id`` already exists."""


class BaseEmbeddingStore(ABC):
    """
    Abstract interface for persisting and retrieving embedding records.

    Implementations must be safe to call repeatedly within a single
    process. Thread-safety across concurrent writers is implementation
    -defined; :class:`JSONEmbeddingStore` documents its own guarantees.
    """

    @abstractmethod
    def add(self, record: EmbeddingRecord) -> None:
        """
        Add a single embedding record to the store.

        Parameters
        ----------
        record:
            The record to persist.

        Raises
        ------
        DuplicateEmbeddingError
            If a record with the same ``embedding_id`` already exists.
        EmbeddingStoreError
            If the record cannot be persisted.
        """
        raise NotImplementedError

    @abstractmethod
    def add_batch(self, records: list[EmbeddingRecord]) -> None:
        """
        Add multiple embedding records to the store in one operation.

        Parameters
        ----------
        records:
            The records to persist. Must not be empty.

        Raises
        ------
        ValueError
            If ``records`` is empty.
        DuplicateEmbeddingError
            If any record's ``embedding_id`` already exists in the store,
            or duplicate IDs exist within ``records`` itself.
        EmbeddingStoreError
            If the records cannot be persisted.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, embedding_id: str) -> EmbeddingRecord:
        """
        Retrieve a single embedding record by ID.

        Parameters
        ----------
        embedding_id:
            The unique ID of the record to retrieve.

        Returns
        -------
        EmbeddingRecord
            The matching record.

        Raises
        ------
        EmbeddingNotFoundError
            If no record with the given ID exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get_by_source(self, source_id: str) -> list[EmbeddingRecord]:
        """
        Retrieve all embedding records derived from a given source document.

        Parameters
        ----------
        source_id:
            The ``source_id`` value (e.g. DOI/PubMed ID) shared by
            :attr:`~embeddings.models.EmbeddingMetadata.source_id` across
            one or more records (e.g. multiple full-text chunks of the
            same paper).

        Returns
        -------
        list[EmbeddingRecord]
            All matching records, ordered by
            :attr:`~embeddings.models.EmbeddingMetadata.chunk_index`.
            Empty list if no records match.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, embedding_id: str) -> None:
        """
        Delete a single embedding record by ID.

        Parameters
        ----------
        embedding_id:
            The unique ID of the record to delete.

        Raises
        ------
        EmbeddingNotFoundError
            If no record with the given ID exists.
        """
        raise NotImplementedError

    @abstractmethod
    def all(self) -> list[EmbeddingRecord]:
        """
        Retrieve every embedding record in the store.

        Returns
        -------
        list[EmbeddingRecord]
            All stored records. Empty list if the store is empty.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of embedding records in the store.

        Returns
        -------
        int
            The number of stored records.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return :meth:`count` so ``len(store)`` works naturally."""
        return self.count()

    def __contains__(self, embedding_id: str) -> bool:
        """Return whether ``embedding_id`` exists in the store."""
        try:
            self.get(embedding_id)
            return True
        except EmbeddingNotFoundError:
            return False


class JSONEmbeddingStore(BaseEmbeddingStore):
    """
    JSON-file-backed implementation of :class:`BaseEmbeddingStore`.

    All records are held in memory (in an ID-keyed dictionary) and
    mirrored to a single JSON file on disk on every mutating operation
    (write-through). This is appropriate for research-scale datasets
    (thousands to low tens-of-thousands of records); for larger corpora,
    a future FAISS- or database-backed :class:`BaseEmbeddingStore`
    implementation should be used instead, without requiring changes to
    any calling code.

    Thread-safety: a single :class:`threading.Lock` guards all mutating
    operations, making this store safe for concurrent use from multiple
    threads within one process. It is not safe for concurrent use from
    multiple processes writing to the same file.

    Parameters
    ----------
    storage_path:
        Path to the JSON file used for persistence. If the file does not
        exist, an empty store is initialized and the file is created on
        the first write. Parent directories are created automatically.

    Examples
    --------
    >>> store = JSONEmbeddingStore(storage_path="data/embeddings.json")
    >>> store.add(record)  # doctest: +SKIP
    >>> retrieved = store.get(record.embedding_id)  # doctest: +SKIP
    """

    def __init__(self, storage_path: str | Path) -> None:
        self._storage_path = Path(storage_path)
        self._lock = threading.Lock()
        self._records: dict[str, EmbeddingRecord] = {}
        self._load()

    @property
    def storage_path(self) -> Path:
        """Return the filesystem path backing this store."""
        return self._storage_path

    def _load(self) -> None:
        """
        Load existing records from ``storage_path`` into memory, if present.

        Raises
        ------
        EmbeddingStoreError
            If the file exists but contains invalid or corrupted data.
        """
        if not self._storage_path.exists():
            logger.info(
                "No existing embedding store found at '%s'; starting empty.",
                self._storage_path,
            )
            return

        try:
            raw_text = self._storage_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                logger.info(
                    "Embedding store file '%s' is empty; starting empty.",
                    self._storage_path,
                )
                return

            raw_records = json.loads(raw_text)
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception(
                "Failed to read embedding store from '%s'.", self._storage_path
            )
            raise EmbeddingStoreError(
                f"Failed to read embedding store from '{self._storage_path}': {exc}"
            ) from exc

        try:
            for raw_record in raw_records:
                record = self._deserialize_record(raw_record)
                self._records[record.embedding_id] = record
        except (KeyError, ValueError, TypeError) as exc:
            logger.exception(
                "Embedding store file '%s' contains malformed data.",
                self._storage_path,
            )
            raise EmbeddingStoreError(
                f"Embedding store file '{self._storage_path}' contains "
                f"malformed data: {exc}"
            ) from exc

        logger.info(
            "Loaded %d embedding record(s) from '%s'.",
            len(self._records),
            self._storage_path,
        )

    def _persist(self) -> None:
        """
        Write all in-memory records to ``storage_path`` as JSON.

        Raises
        ------
        EmbeddingStoreError
            If the file cannot be written.
        """
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = [
                self._serialize_record(record)
                for record in self._records.values()
            ]
            tmp_path = self._storage_path.with_suffix(
                self._storage_path.suffix + ".tmp"
            )
            tmp_path.write_text(
                json.dumps(serialized, indent=2), encoding="utf-8"
            )
            tmp_path.replace(self._storage_path)
        except OSError as exc:
            logger.exception(
                "Failed to write embedding store to '%s'.", self._storage_path
            )
            raise EmbeddingStoreError(
                f"Failed to write embedding store to '{self._storage_path}': {exc}"
            ) from exc

    @staticmethod
    def _serialize_record(record: EmbeddingRecord) -> dict:
        """Convert an :class:`EmbeddingRecord` into a JSON-serializable dict."""
        return {
            "embedding_id": record.embedding_id,
            "vector": record.vector,
            "model_name": record.model_name,
            "created_at": record.created_at.isoformat(),
            "metadata": {
                "source_id": record.metadata.source_id,
                "source_type": record.metadata.source_type,
                "title": record.metadata.title,
                "text": record.metadata.text,
                "chunk_index": record.metadata.chunk_index,
                "extra": record.metadata.extra,
            },
        }

    @staticmethod
    def _deserialize_record(raw: dict) -> EmbeddingRecord:
        """Convert a JSON dict back into an :class:`EmbeddingRecord`."""
        raw_metadata = raw["metadata"]
        metadata = EmbeddingMetadata(
            source_id=raw_metadata["source_id"],
            source_type=raw_metadata["source_type"],
            title=raw_metadata["title"],
            text=raw_metadata["text"],
            chunk_index=raw_metadata.get("chunk_index", 0),
            extra=raw_metadata.get("extra", {}),
        )
        return EmbeddingRecord(
            metadata=metadata,
            vector=raw["vector"],
            model_name=raw["model_name"],
            embedding_id=raw["embedding_id"],
            created_at=datetime.fromisoformat(raw["created_at"]),
        )

    def add(self, record: EmbeddingRecord) -> None:
        """Add a single embedding record. See :meth:`BaseEmbeddingStore.add`."""
        with self._lock:
            if record.embedding_id in self._records:
                raise DuplicateEmbeddingError(
                    f"Embedding with id '{record.embedding_id}' already "
                    f"exists in the store."
                )
            self._records[record.embedding_id] = record
            self._persist()
            logger.debug("Added embedding record '%s'.", record.embedding_id)

    def add_batch(self, records: list[EmbeddingRecord]) -> None:
        """Add multiple records. See :meth:`BaseEmbeddingStore.add_batch`."""
        if not records:
            raise ValueError("records must not be empty.")

        with self._lock:
            seen_ids: set[str] = set()
            for record in records:
                if record.embedding_id in self._records:
                    raise DuplicateEmbeddingError(
                        f"Embedding with id '{record.embedding_id}' already "
                        f"exists in the store."
                    )
                if record.embedding_id in seen_ids:
                    raise DuplicateEmbeddingError(
                        f"Duplicate embedding id '{record.embedding_id}' "
                        f"within the provided batch."
                    )
                seen_ids.add(record.embedding_id)

            for record in records:
                self._records[record.embedding_id] = record

            self._persist()
            logger.debug("Added %d embedding record(s) in batch.", len(records))

    def get(self, embedding_id: str) -> EmbeddingRecord:
        """Retrieve a record by ID. See :meth:`BaseEmbeddingStore.get`."""
        with self._lock:
            record = self._records.get(embedding_id)
            if record is None:
                raise EmbeddingNotFoundError(
                    f"No embedding found with id '{embedding_id}'."
                )
            return record

    def get_by_source(self, source_id: str) -> list[EmbeddingRecord]:
        """Retrieve records by source. See :meth:`BaseEmbeddingStore.get_by_source`."""
        with self._lock:
            matches = [
                record
                for record in self._records.values()
                if record.metadata.source_id == source_id
            ]
        return sorted(matches, key=lambda r: r.metadata.chunk_index)

    def delete(self, embedding_id: str) -> None:
        """Delete a record by ID. See :meth:`BaseEmbeddingStore.delete`."""
        with self._lock:
            if embedding_id not in self._records:
                raise EmbeddingNotFoundError(
                    f"No embedding found with id '{embedding_id}'."
                )
            del self._records[embedding_id]
            self._persist()
            logger.debug("Deleted embedding record '%s'.", embedding_id)

    def all(self) -> list[EmbeddingRecord]:
        """Return all records. See :meth:`BaseEmbeddingStore.all`."""
        with self._lock:
            return list(self._records.values())

    def count(self) -> int:
        """Return the number of stored records. See :meth:`BaseEmbeddingStore.count`."""
        with self._lock:
            return len(self._records)


__all__: list[str] = [
    "EmbeddingStoreError",
    "EmbeddingNotFoundError",
    "DuplicateEmbeddingError",
    "BaseEmbeddingStore",
    "JSONEmbeddingStore",
]