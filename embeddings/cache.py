"""
Embedding cache for the BioNexus Embeddings module.

Generating embeddings (especially via a neural model) is comparatively
expensive. This module provides a cache keyed on ``(model_name, text)``
pairs so that repeated requests to embed identical text — a common
occurrence when the same query or paper abstract is processed multiple
times across searches, re-indexing runs, or RAG calls — can be served
from memory instead of re-invoking the embedding model.

The cache is deliberately kept independent of
:mod:`embeddings.embedding_generator`: it caches raw vectors keyed by a
string cache key (computed via :func:`embeddings.utils.compute_cache_key`),
and knows nothing about ``SentenceTransformer`` or any other backend. This
keeps it reusable and independently testable.

Classes
-------
BaseEmbeddingCache
    Abstract interface for an embedding vector cache.
LRUEmbeddingCache
    Concrete, thread-safe, in-memory Least-Recently-Used cache with an
    optional time-to-live (TTL) per entry.

Exceptions
----------
EmbeddingCacheError
    Raised for cache-layer failures.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EmbeddingCacheError(RuntimeError):
    """Raised when the embedding cache encounters an unrecoverable error."""


@dataclass(frozen=True)
class CacheStats:
    """
    Snapshot of cache performance statistics.

    Attributes
    ----------
    hits:
        Number of successful cache lookups (:meth:`BaseEmbeddingCache.get`
        calls that returned a cached vector).
    misses:
        Number of cache lookups that found no valid entry.
    evictions:
        Number of entries removed due to capacity limits (LRU eviction).
    expirations:
        Number of entries removed because their TTL elapsed.
    size:
        Current number of entries held in the cache.
    """

    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int

    @property
    def hit_rate(self) -> float:
        """
        Return the cache hit rate as a fraction in ``[0.0, 1.0]``.

        Returns ``0.0`` if no lookups have been performed yet, to avoid a
        division-by-zero error.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class BaseEmbeddingCache(ABC):
    """
    Abstract interface for a cache of embedding vectors.

    Implementations map string cache keys (typically produced by
    :func:`embeddings.utils.compute_cache_key`) to embedding vectors,
    supporting bounded memory usage and optional expiration.
    """

    @abstractmethod
    def get(self, key: str) -> list[float] | None:
        """
        Retrieve a cached vector by key.

        Parameters
        ----------
        key:
            The cache key to look up.

        Returns
        -------
        list[float] | None
            The cached vector if present and not expired, otherwise
            ``None``.

        Raises
        ------
        ValueError
            If ``key`` is empty.
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, key: str, vector: list[float]) -> None:
        """
        Store a vector in the cache under the given key.

        Parameters
        ----------
        key:
            The cache key to store the vector under.
        vector:
            The embedding vector to cache.

        Raises
        ------
        ValueError
            If ``key`` is empty or ``vector`` is empty.
        """
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """
        Remove a single entry from the cache, if present.

        Parameters
        ----------
        key:
            The cache key to remove. Silently no-ops if the key is not
            present.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache and reset statistics."""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> CacheStats:
        """
        Return current cache performance statistics.

        Returns
        -------
        CacheStats
            A snapshot of hits, misses, evictions, expirations, and size.
        """
        raise NotImplementedError

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` currently has a valid (non-expired) entry."""
        return self.get(key) is not None


class LRUEmbeddingCache(BaseEmbeddingCache):
    """
    Thread-safe, in-memory LRU cache for embedding vectors.

    Entries are evicted in least-recently-used order once
    ``max_size`` is exceeded. An optional ``ttl_seconds`` causes entries
    older than the TTL to be treated as expired (removed lazily, on
    access).

    Parameters
    ----------
    max_size:
        Maximum number of entries to retain. Must be positive. When a
        ``put`` would exceed this size, the least-recently-used entry is
        evicted.
    ttl_seconds:
        Optional time-to-live, in seconds, for each entry. If ``None``,
        entries never expire based on age (only LRU eviction applies).
        If provided, must be positive.

    Examples
    --------
    >>> cache = LRUEmbeddingCache(max_size=2)
    >>> cache.put("a", [1.0, 2.0])
    >>> cache.get("a")
    [1.0, 2.0]
    >>> cache.put("b", [3.0, 4.0])
    >>> cache.put("c", [5.0, 6.0])  # evicts "a" (least recently used)
    >>> cache.get("a") is None
    True
    """

    def __init__(
        self,
        max_size: int = 10_000,
        ttl_seconds: float | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}.")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive when provided, got {ttl_seconds}."
            )

        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, tuple[list[float], float]]" = OrderedDict()
        self._lock = threading.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    @property
    def max_size(self) -> int:
        """Return the maximum number of entries this cache will retain."""
        return self._max_size

    @property
    def ttl_seconds(self) -> float | None:
        """Return the configured entry time-to-live in seconds, if any."""
        return self._ttl_seconds

    def get(self, key: str) -> list[float] | None:
        """Retrieve a cached vector by key. See :meth:`BaseEmbeddingCache.get`."""
        if not key or not key.strip():
            raise ValueError("key must be a non-empty string.")

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            vector, stored_at = entry

            if self._is_expired(stored_at):
                del self._entries[key]
                self._misses += 1
                self._expirations += 1
                logger.debug("Cache entry '%s' expired and was removed.", key)
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return list(vector)

    def put(self, key: str, vector: list[float]) -> None:
        """Store a vector under a key. See :meth:`BaseEmbeddingCache.put`."""
        if not key or not key.strip():
            raise ValueError("key must be a non-empty string.")
        if not vector:
            raise ValueError("vector must not be empty.")

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)

            self._entries[key] = (list(vector), time.monotonic())

            while len(self._entries) > self._max_size:
                evicted_key, _ = self._entries.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    "Evicted least-recently-used cache entry '%s' "
                    "(max_size=%d exceeded).",
                    evicted_key,
                    self._max_size,
                )

    def invalidate(self, key: str) -> None:
        """Remove a single entry. See :meth:`BaseEmbeddingCache.invalidate`."""
        if not key or not key.strip():
            raise ValueError("key must be a non-empty string.")

        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
            logger.debug("Embedding cache cleared.")

    def stats(self) -> CacheStats:
        """Return current cache performance statistics."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._entries),
            )

    def _is_expired(self, stored_at: float) -> bool:
        """Return whether an entry stored at ``stored_at`` has exceeded the TTL."""
        if self._ttl_seconds is None:
            return False
        return (time.monotonic() - stored_at) > self._ttl_seconds

    def __len__(self) -> int:
        """Return the current number of entries in the cache."""
        with self._lock:
            return len(self._entries)


__all__: list[str] = [
    "EmbeddingCacheError",
    "CacheStats",
    "BaseEmbeddingCache",
    "LRUEmbeddingCache",
]