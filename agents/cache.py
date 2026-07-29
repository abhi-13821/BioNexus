"""
agents/cache.py

Reusable cache system for the BioNexus Multi-Agent AI System.

This module provides a thread-safe, in-memory cache with TTL support,
LRU eviction, namespacing, and statistics tracking. It is designed to
be used across all agents for caching responses, execution plans,
conversation metadata, and other frequently accessed data.

Core capabilities:
    - Thread-safe operations with fine-grained locking
    - TTL (Time-To-Live) support with automatic expiration
    - LRU (Least Recently Used) eviction with configurable max size
    - Namespace support for organized cache isolation
    - Cache statistics (hits, misses, size, evictions, expirations)
    - Cache invalidation (single key, namespace, or all)
    - Serialization helpers for complex objects
    - Configurable default TTL and max size

Compatibility
-------------
Targets Python 3.11. Uses only the standard library.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variables
K = TypeVar("K")
V = TypeVar("V")


# ----------------------------------------------------------------------
# Cache exceptions
# ----------------------------------------------------------------------


class CacheError(Exception):
    """Base exception for cache operations."""

    pass


class CacheKeyError(CacheError):
    """Raised when a cache key is invalid."""

    pass


class CacheSerializationError(CacheError):
    """Raised when serialization/deserialization fails."""

    pass


class CacheNamespaceError(CacheError):
    """Raised when a namespace operation fails."""

    pass


# ----------------------------------------------------------------------
# Cache configuration
# ----------------------------------------------------------------------


@dataclass
class CacheConfig:
    """
    Configuration for the cache system.

    Attributes:
        max_size: Maximum number of entries per namespace.
        default_ttl_seconds: Default TTL in seconds for entries.
        track_stats: Whether to track hit/miss statistics.
        enable_serialization: Whether to enable serialization.
        namespace_max_size: Optional per-namespace max size override.
        namespace_ttl: Optional per-namespace TTL override.
    """

    max_size: int = 1000
    default_ttl_seconds: float = 300.0  # 5 minutes
    track_stats: bool = True
    enable_serialization: bool = True
    namespace_max_size: dict[str, int] = field(default_factory=dict)
    namespace_ttl: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_size < 1:
            raise ValueError("max_size must be positive")
        if self.default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be non-negative")
        for ns, size in self.namespace_max_size.items():
            if size < 1:
                raise ValueError(f"namespace_max_size for '{ns}' must be positive")
        for ns, ttl in self.namespace_ttl.items():
            if ttl < 0:
                raise ValueError(f"namespace_ttl for '{ns}' must be non-negative")


# ----------------------------------------------------------------------
# Cache entry
# ----------------------------------------------------------------------


@dataclass
class CacheEntry(Generic[V]):
    """
    A single cache entry with metadata.

    Attributes:
        value: The cached value.
        namespace: The namespace of the entry.
        created_at: Timestamp when the entry was created.
        expires_at: Timestamp when the entry expires (None for no expiry).
        last_accessed: Timestamp of last access.
        hit_count: Number of times this entry has been accessed.
        access_count: Total number of accesses (including misses for this key).
        size_bytes: Approximate size of the value in bytes.
    """

    value: V
    namespace: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """
        Check if the entry has expired.

        Args:
            now: Current time. Defaults to now.

        Returns:
            True if expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now >= self.expires_at

    def touch(self, now: Optional[datetime] = None) -> None:
        """
        Update the last accessed timestamp.

        Args:
            now: Current time. Defaults to now.
        """
        now = now or datetime.now(timezone.utc)
        self.last_accessed = now
        self.hit_count += 1


# ----------------------------------------------------------------------
# Cache statistics
# ----------------------------------------------------------------------


@dataclass
class NamespaceStatistics:
    """
    Cache statistics for a single namespace.

    Attributes:
        namespace: The namespace name.
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of LRU evictions.
        expirations: Number of TTL expirations.
        entry_count: Current number of entries.
        total_size_bytes: Approximate total size in bytes.
    """

    namespace: str
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    entry_count: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheStatistics:
    """
    Aggregate cache statistics.

    Attributes:
        total_hits: Total hits across all namespaces.
        total_misses: Total misses across all namespaces.
        total_evictions: Total evictions across all namespaces.
        total_expirations: Total expirations across all namespaces.
        total_entries: Total entries across all namespaces.
        total_size_bytes: Total size in bytes across all namespaces.
        namespace_stats: Per-namespace statistics.
    """

    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    namespace_stats: dict[str, NamespaceStatistics] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate the overall hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


# ----------------------------------------------------------------------
# Cache implementation
# ----------------------------------------------------------------------


class Cache(Generic[V]):
    """
    Thread-safe, in-memory cache with TTL and LRU eviction.

    This cache supports namespacing, TTL expiration, LRU eviction,
    and statistics tracking.

    Example:
        >>> cache = Cache()
        >>> cache.set("user:123", {"name": "Alice"})
        >>> value = cache.get("user:123")
        >>> print(value["name"])
        Alice

    Attributes:
        config: The cache configuration.
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """
        Initialize the cache.

        Args:
            config: Cache configuration. Defaults to CacheConfig().
        """
        self.config: CacheConfig = config or CacheConfig()
        self._lock: threading.RLock = threading.RLock()
        self._entries: dict[str, CacheEntry[V]] = OrderedDict()
        self._namespace_entries: dict[str, OrderedDict[str, CacheEntry[V]]] = {}
        self._namespace_stats: dict[str, NamespaceStatistics] = {}
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0
        self._total_expirations: int = 0
        self._logger: logging.Logger = logging.getLogger(f"{__name__}.Cache")

    def _get_namespace(self, key: str) -> str:
        """
        Extract the namespace from a key.

        Args:
            key: The cache key.

        Returns:
            The namespace.
        """
        if ":" in key:
            return key.split(":", 1)[0]
        return "default"

    def _get_namespace_max_size(self, namespace: str) -> int:
        """
        Get the max size for a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The max size.
        """
        return self.config.namespace_max_size.get(namespace, self.config.max_size)

    def _get_namespace_ttl(self, namespace: str) -> float:
        """
        Get the TTL for a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The TTL in seconds.
        """
        return self.config.namespace_ttl.get(namespace, self.config.default_ttl_seconds)

    def _get_namespace_entries(self, namespace: str) -> OrderedDict[str, CacheEntry[V]]:
        """
        Get or create the ordered dict for a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The ordered dict of entries.
        """
        if namespace not in self._namespace_entries:
            self._namespace_entries[namespace] = OrderedDict()
        return self._namespace_entries[namespace]

    def _get_namespace_stats(self, namespace: str) -> NamespaceStatistics:
        """
        Get or create statistics for a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The namespace statistics.
        """
        if namespace not in self._namespace_stats:
            self._namespace_stats[namespace] = NamespaceStatistics(namespace=namespace)
        return self._namespace_stats[namespace]

    def _evict_lru(self, namespace: str) -> None:
        """
        Evict the least recently used entry from a namespace.

        Args:
            namespace: The namespace name.
        """
        entries = self._get_namespace_entries(namespace)
        if not entries:
            return

        # Remove the oldest entry (first in OrderedDict)
        key, entry = entries.popitem(last=False)

        # Remove from main dictionary
        if key in self._entries:
            del self._entries[key]

        # Update statistics
        self._total_evictions += 1
        stats = self._get_namespace_stats(namespace)
        stats.evictions += 1

        self._logger.debug("Evicted LRU entry '%s' from namespace '%s'", key, namespace)

    def _evict_expired(self, namespace: str, now: Optional[datetime] = None) -> int:
        """
        Evict expired entries from a namespace.

        Args:
            namespace: The namespace name.
            now: Current time.

        Returns:
            Number of entries evicted.
        """
        now = now or datetime.now(timezone.utc)
        entries = self._get_namespace_entries(namespace)
        keys_to_remove = []

        for key, entry in entries.items():
            if entry.is_expired(now):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            # Remove from namespace
            del entries[key]

            # Remove from main dictionary
            if key in self._entries:
                del self._entries[key]

            # Update statistics
            self._total_expirations += 1
            stats = self._get_namespace_stats(namespace)
            stats.expirations += 1

            self._logger.debug("Expired entry '%s' from namespace '%s'", key, namespace)

        return len(keys_to_remove)

    def _get_max_size(self, namespace: str) -> int:
        """Get the max size for a namespace."""
        return self.config.namespace_max_size.get(namespace, self.config.max_size)

    def _get_ttl(self, namespace: str) -> float:
        """Get the TTL for a namespace."""
        return self.config.namespace_ttl.get(namespace, self.config.default_ttl_seconds)

    def set(self, key: str, value: V, ttl: Optional[float] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: TTL in seconds for this entry. Overrides default TTL.

        Raises:
            CacheKeyError: If the key is invalid.
        """
        if not key or not key.strip():
            raise CacheKeyError("Key must be a non-empty string")

        namespace = self._get_namespace(key)
        effective_ttl = ttl if ttl is not None else self._get_ttl(namespace)

        with self._lock:
            # Evict expired entries
            self._evict_expired(namespace)

            # Check if we need to evict LRU
            entries = self._get_namespace_entries(namespace)
            max_size = self._get_max_size(namespace)

            if len(entries) >= max_size:
                self._evict_lru(namespace)

            # Create entry
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(seconds=effective_ttl) if effective_ttl > 0 else None

            # Calculate approximate size
            size_bytes = len(str(value)) if value is not None else 0

            entry = CacheEntry(
                value=value,
                namespace=namespace,
                created_at=now,
                expires_at=expires_at,
                last_accessed=now,
                size_bytes=size_bytes,
            )

            # Store entry
            self._entries[key] = entry

            # If key exists in namespace, remove and re-add (update order)
            if key in entries:
                del entries[key]

            entries[key] = entry

            self._logger.debug("Cached entry '%s' in namespace '%s' (ttl=%s)", key, namespace, effective_ttl)

    def get(self, key: str, default: Optional[V] = None) -> Optional[V]:
        """
        Retrieve a value from the cache.

        Args:
            key: The cache key.
            default: Default value if key is not found.

        Returns:
            The cached value, or default if not found.
        """
        if not key or not key.strip():
            return default

        namespace = self._get_namespace(key)

        with self._lock:
            # Check if entry exists
            if key not in self._entries:
                if self.config.track_stats:
                    self._total_misses += 1
                    stats = self._get_namespace_stats(namespace)
                    stats.misses += 1
                return default

            entry = self._entries[key]

            # Check if expired
            now = datetime.now(timezone.utc)
            if entry.is_expired(now):
                # Remove expired entry
                self._evict_expired(namespace, now)
                if self.config.track_stats:
                    self._total_misses += 1
                    stats = self._get_namespace_stats(namespace)
                    stats.misses += 1
                return default

            # Update access
            entry.touch(now)
            self._entries[key] = entry

            # Update order in namespace
            entries = self._get_namespace_entries(namespace)
            if key in entries:
                del entries[key]
                entries[key] = entry

            if self.config.track_stats:
                self._total_hits += 1
                stats = self._get_namespace_stats(namespace)
                stats.hits += 1

            self._logger.debug("Cache hit for '%s' in namespace '%s'", key, namespace)

            return entry.value

    def get_with_metadata(self, key: str) -> tuple[Optional[V], Optional[CacheEntry[V]]]:
        """
        Retrieve a value with its metadata.

        Args:
            key: The cache key.

        Returns:
            A tuple of (value, entry) or (None, None) if not found.
        """
        if not key or not key.strip():
            return None, None

        with self._lock:
            if key not in self._entries:
                return None, None

            entry = self._entries[key]

            # Check if expired
            now = datetime.now(timezone.utc)
            if entry.is_expired(now):
                namespace = self._get_namespace(key)
                self._evict_expired(namespace, now)
                return None, None

            return entry.value, entry

    def contains(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: The cache key.

        Returns:
            True if the key exists and is valid.
        """
        if not key or not key.strip():
            return False

        with self._lock:
            if key not in self._entries:
                return False

            entry = self._entries[key]
            if entry.is_expired():
                namespace = self._get_namespace(key)
                self._evict_expired(namespace)
                return False

            return True

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a single cache entry.

        Args:
            key: The cache key.

        Returns:
            True if the entry was removed, False otherwise.
        """
        if not key or not key.strip():
            return False

        with self._lock:
            if key not in self._entries:
                return False

            namespace = self._get_namespace(key)

            # Remove from main dictionary
            del self._entries[key]

            # Remove from namespace
            entries = self._get_namespace_entries(namespace)
            if key in entries:
                del entries[key]

            self._logger.debug("Invalidated entry '%s'", key)

            return True

    def invalidate_namespace(self, namespace: str) -> int:
        """
        Invalidate all entries in a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The number of entries removed.
        """
        if not namespace or not namespace.strip():
            raise CacheKeyError("Namespace must be a non-empty string")

        with self._lock:
            if namespace not in self._namespace_entries:
                return 0

            entries = self._namespace_entries[namespace]
            keys_to_remove = list(entries.keys())

            for key in keys_to_remove:
                if key in self._entries:
                    del self._entries[key]

            entries.clear()

            self._logger.debug("Invalidated namespace '%s' (%d entries)", namespace, len(keys_to_remove))

            return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            The number of entries removed.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._namespace_entries.clear()

            self._logger.debug("Cleared cache (%d entries)", count)

            return count

    def get_stats(self) -> CacheStatistics:
        """
        Get cache statistics.

        Returns:
            A CacheStatistics object.
        """
        with self._lock:
            namespace_stats: dict[str, NamespaceStatistics] = {}

            for namespace, entries in self._namespace_entries.items():
                stats = self._get_namespace_stats(namespace)
                stats.entry_count = len(entries)
                stats.total_size_bytes = sum(e.size_bytes for e in entries.values())
                namespace_stats[namespace] = stats

            return CacheStatistics(
                total_hits=self._total_hits,
                total_misses=self._total_misses,
                total_evictions=self._total_evictions,
                total_expirations=self._total_expirations,
                total_entries=len(self._entries),
                total_size_bytes=sum(e.size_bytes for e in self._entries.values()),
                namespace_stats=namespace_stats,
            )

    def get_namespace_size(self, namespace: str) -> int:
        """
        Get the number of entries in a namespace.

        Args:
            namespace: The namespace name.

        Returns:
            The number of entries.
        """
        with self._lock:
            if namespace not in self._namespace_entries:
                return 0
            return len(self._namespace_entries[namespace])

    def get_keys(self, namespace: Optional[str] = None) -> list[str]:
        """
        Get all keys in the cache, optionally filtered by namespace.

        Args:
            namespace: Optional namespace filter.

        Returns:
            A list of keys.
        """
        with self._lock:
            if namespace is None:
                return list(self._entries.keys())

            if namespace not in self._namespace_entries:
                return []

            return list(self._namespace_entries[namespace].keys())

    def prune_expired(self) -> int:
        """
        Prune all expired entries from the cache.

        Returns:
            The number of entries removed.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            total_removed = 0

            for namespace in list(self._namespace_entries.keys()):
                removed = self._evict_expired(namespace, now)
                total_removed += removed

            return total_removed


# ----------------------------------------------------------------------
# Cache decorator
# ----------------------------------------------------------------------


def cached(
    ttl: Optional[float] = None,
    namespace: Optional[str] = None,
    cache: Optional[Cache] = None,
    key_builder: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., V]], Callable[..., V]]:
    """
    Decorator that caches function results.

    Args:
        ttl: TTL for cached results in seconds.
        namespace: Cache namespace. Uses function name if not provided.
        cache: Cache instance. Uses default cache if not provided.
        key_builder: Function that builds a cache key from arguments.

    Returns:
        A decorated function.
    """
    _default_cache: Optional[Cache] = None

    def get_cache() -> Cache:
        nonlocal _default_cache
        if _default_cache is None:
            _default_cache = Cache()
        return cache or _default_cache

    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> V:
            active_cache = get_cache()
            ns = namespace or func.__name__
            builder = key_builder or default_key_builder

            key = builder(ns, *args, **kwargs)

            # Try to get from cache
            cached_value = active_cache.get(key)
            if cached_value is not None:
                return cached_value  # type: ignore

            # Compute and cache
            result = func(*args, **kwargs)

            # Only cache non-None results
            if result is not None:
                active_cache.set(key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def default_key_builder(namespace: str, *args: Any, **kwargs: Any) -> str:
    """
    Build a cache key from arguments.

    Args:
        namespace: The namespace.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        A cache key string.
    """
    parts = [namespace]

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            parts.append(str(arg))
        elif isinstance(arg, Enum):
            parts.append(arg.value)
        elif hasattr(arg, "__dict__"):
            parts.append(str(json.dumps(arg.__dict__, sort_keys=True, default=str)))
        else:
            parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
        elif isinstance(v, Enum):
            parts.append(f"{k}={v.value}")
        else:
            parts.append(f"{k}={str(v)}")

    combined = "|".join(parts)
    return f"{namespace}:{hashlib.md5(combined.encode()).hexdigest()[:16]}"


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


def serialize_value(value: Any) -> str:
    """
    Serialize a value to a string.

    Args:
        value: The value to serialize.

    Returns:
        A JSON string.

    Raises:
        CacheSerializationError: If serialization fails.
    """
    try:
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value, default=str)
        return json.dumps(value, default=str)
    except (TypeError, ValueError) as e:
        raise CacheSerializationError(f"Failed to serialize value: {e}") from e


def deserialize_value(data: str) -> Any:
    """
    Deserialize a value from a string.

    Args:
        data: The serialized data.

    Returns:
        The deserialized value.

    Raises:
        CacheSerializationError: If deserialization fails.
    """
    try:
        return json.loads(data)
    except (TypeError, ValueError) as e:
        raise CacheSerializationError(f"Failed to deserialize value: {e}") from e


# ----------------------------------------------------------------------
# Module-level cache
# ----------------------------------------------------------------------

_default_cache: Optional[Cache] = None


def get_default_cache() -> Cache:
    """
    Get the default cache instance.

    Returns:
        The default cache.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


def clear_default_cache() -> None:
    """Clear the default cache."""
    global _default_cache
    if _default_cache is not None:
        _default_cache.clear()


def reset_default_cache(config: Optional[CacheConfig] = None) -> Cache:
    """
    Reset the default cache with new configuration.

    Args:
        config: New cache configuration.

    Returns:
        The new default cache.
    """
    global _default_cache
    _default_cache = Cache(config=config)
    return _default_cache


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__: list[str] = [
    "CacheError",
    "CacheKeyError",
    "CacheSerializationError",
    "CacheNamespaceError",
    "CacheConfig",
    "CacheEntry",
    "NamespaceStatistics",
    "CacheStatistics",
    "Cache",
    "cached",
    "default_key_builder",
    "serialize_value",
    "deserialize_value",
    "get_default_cache",
    "clear_default_cache",
    "reset_default_cache",
]