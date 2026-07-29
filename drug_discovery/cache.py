"""
drug_discovery/cache.py

Thread-safe in-memory caching for expensive Drug Discovery computations.

This module provides a generic, dependency-free caching layer intended
to sit in front of the costly operations performed elsewhere in the
Drug Discovery module -- molecular property calculation
(``molecular_properties``), compound similarity search
(``compound_similarity``), target prediction (``target_prediction``),
toxicity prediction (``toxicity_prediction``), drug repurposing
(``drug_repurposing``), and candidate ranking (``candidate_ranking``).

Core capabilities:
    - A thread-safe ``LRUTTLCache`` combining size-bounded LRU eviction
      with optional per-entry time-to-live (TTL) expiration.
    - Deterministic cache key generation from arbitrary function
      arguments, including dataclasses and enums from
      ``drug_discovery.models``.
    - Namespaced cache statistics (hits, misses, evictions,
      expirations) both globally and per module/namespace.
    - Namespace-scoped and whole-cache invalidation/clearing.
    - A ``cached`` decorator (plus thin per-module convenience wrappers)
      so any function in the other Drug Discovery modules can opt into
      caching with a single line.
    - Best-effort serialization of cache contents (including
      dataclasses) for diagnostics, logging, or cross-process snapshots.

This module has no dependency on RDKit or any other Drug Discovery
module, so it can be imported freely from all of them without
introducing circular imports. Cache keys should be built from simple,
stable inputs (SMILES strings, compound IDs, disease names, numeric
parameters) rather than live objects such as RDKit ``Mol`` instances,
whose identity is not stable across calls.

Compatibility
-------------
Targets Python 3.11. Uses only the standard library.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

__all__ = [
    "CacheError",
    "CacheKeyError",
    "CacheSerializationError",
    "CacheNamespace",
    "CacheConfig",
    "CacheEntry",
    "NamespaceStatistics",
    "CacheStatistics",
    "LRUTTLCache",
    "make_cache_key",
    "cached",
    "get_default_cache",
    "configure_default_cache",
    "reset_default_cache",
    "cache_molecular_properties",
    "cache_compound_similarity",
    "cache_target_prediction",
    "cache_toxicity_prediction",
    "cache_drug_repurposing",
    "cache_candidate_ranking",
]

logger = logging.getLogger(__name__)

_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CacheError(RuntimeError):
    """Base exception for unexpected cache operation failures."""


class CacheKeyError(ValueError):
    """
    Raised when a stable cache key cannot be generated from the given
    namespace, positional arguments, or keyword arguments.
    """


class CacheSerializationError(CacheError):
    """
    Raised when cache contents cannot be serialized into a snapshot
    (e.g., ``export_snapshot``) using the configured or default
    serializer.
    """


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class CacheNamespace(str, Enum):
    """
    Standard cache namespaces corresponding to each cacheable Drug
    Discovery module. Using these constants (rather than free-form
    strings) keeps cache keys and statistics consistent across the
    module, while still allowing callers to define their own namespace
    strings for custom or future use cases.
    """

    MOLECULAR_PROPERTIES = "molecular_properties"
    COMPOUND_SIMILARITY = "compound_similarity"
    TARGET_PREDICTION = "target_prediction"
    TOXICITY_PREDICTION = "toxicity_prediction"
    DRUG_REPURPOSING = "drug_repurposing"
    CANDIDATE_RANKING = "candidate_ranking"
    GENERIC = "generic"


# ---------------------------------------------------------------------------
# Configuration and entry model
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CacheConfig:
    """
    Configuration governing the size, expiration, and behavior of an
    ``LRUTTLCache``.

    Attributes:
        max_size: Maximum number of entries the cache may hold before
            least-recently-used entries are evicted.
        default_ttl_seconds: Default time-to-live, in seconds, applied to
            entries that do not specify their own TTL. ``None`` means
            entries never expire by default.
        track_namespace_stats: Whether to additionally track hit/miss/
            eviction/expiration statistics broken down by namespace (the
            prefix of each cache key before the first colon).
    """

    max_size: int = 2048
    default_ttl_seconds: float | None = None
    track_namespace_stats: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_size < 1:
            raise ValueError("max_size must be a positive integer.")
        if self.default_ttl_seconds is not None and self.default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive when provided.")


@dataclasses.dataclass
class CacheEntry:
    """
    A single stored cache entry, including bookkeeping metadata used for
    TTL expiration and LRU/statistics reporting.

    Attributes:
        value: The cached value.
        created_at: UTC timestamp when the entry was stored.
        expires_at: UTC timestamp after which the entry is considered
            expired, or ``None`` if the entry never expires.
        last_accessed_at: UTC timestamp of the most recent cache hit for
            this entry.
        hit_count: Number of times this entry has been retrieved via a
            cache hit.
    """

    value: Any
    created_at: datetime
    expires_at: datetime | None
    last_accessed_at: datetime
    hit_count: int = 0

    def is_expired(self, now: datetime | None = None) -> bool:
        """
        Determine whether this entry has expired.

        Args:
            now: The current time to compare against. Defaults to the
                current UTC time if omitted.

        Returns:
            True if ``expires_at`` is set and is at or before ``now``,
            False otherwise.
        """
        if self.expires_at is None:
            return False
        return self.expires_at <= (now or datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NamespaceStatistics:
    """
    Cache statistics scoped to a single namespace.

    Attributes:
        namespace: The namespace these statistics describe.
        hits: Number of cache hits recorded for this namespace.
        misses: Number of cache misses recorded for this namespace.
        evictions: Number of LRU evictions recorded for this namespace.
        expirations: Number of TTL expirations recorded for this
            namespace.
        entry_count: Number of entries currently stored for this
            namespace.
    """

    namespace: str
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """
        Return the cache hit rate for this namespace.

        Returns:
            ``hits / (hits + misses)``, or 0.0 if no lookups have been
            recorded.
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclasses.dataclass
class CacheStatistics:
    """
    Aggregate statistics describing a cache's overall behavior.

    Attributes:
        total_hits: Total number of cache hits across all namespaces.
        total_misses: Total number of cache misses across all
            namespaces.
        total_evictions: Total number of LRU evictions across all
            namespaces.
        total_expirations: Total number of TTL expirations across all
            namespaces.
        current_entry_count: Number of entries currently stored.
        max_size: The cache's configured maximum size.
        namespace_stats: Per-namespace statistics breakdown.
    """

    total_hits: int
    total_misses: int
    total_evictions: int
    total_expirations: int
    current_entry_count: int
    max_size: int
    namespace_stats: dict[str, NamespaceStatistics] = dataclasses.field(
        default_factory=dict
    )

    @property
    def hit_rate(self) -> float:
        """
        Return the overall cache hit rate.

        Returns:
            ``total_hits / (total_hits + total_misses)``, or 0.0 if no
            lookups have been recorded.
        """
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    @property
    def fill_ratio(self) -> float:
        """
        Return how full the cache currently is, relative to its maximum
        size.

        Returns:
            ``current_entry_count / max_size``, in [0.0, 1.0].
        """
        return self.current_entry_count / self.max_size if self.max_size else 0.0


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


def _stable_value_repr(value: Any) -> Any:
    """
    Convert an arbitrary value into a deterministic, JSON-serializable
    representation suitable for hashing into a cache key.

    Handles primitives, ``Enum`` members, dataclass instances (including
    those defined in ``drug_discovery.models``), mappings, and
    sequences/sets recursively. Any other object type falls back to its
    ``repr()``.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable representation of ``value``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": value.value}

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            "fields": {
                field.name: _stable_value_repr(getattr(value, field.name))
                for field in sorted(
                    dataclasses.fields(value), key=lambda f: f.name
                )
            },
        }

    if isinstance(value, dict):
        return {
            "__mapping__": True,
            "items": [
                [_stable_value_repr(key), _stable_value_repr(item)]
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            ],
        }

    if isinstance(value, (list, tuple)):
        return [_stable_value_repr(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return sorted(_stable_value_repr(item) for item in value)

    return repr(value)


def make_cache_key(namespace: str, *args: Any, **kwargs: Any) -> str:
    """
    Generate a deterministic cache key for a namespace and a set of
    positional/keyword arguments.

    The key is stable across process runs for equivalent inputs: the
    same namespace, positional arguments (in order), and keyword
    arguments (regardless of the order they were passed in) always
    produce the same key. Supports primitives, ``Enum`` members,
    dataclasses (including ``drug_discovery.models`` types), and nested
    mappings/sequences/sets.

    Args:
        namespace: A short identifier grouping related cache entries
            (typically a ``CacheNamespace`` value or module name).
        *args: Positional arguments to incorporate into the key.
        **kwargs: Keyword arguments to incorporate into the key.

    Returns:
        A cache key string of the form ``"<namespace>:<hex_digest>"``.

    Raises:
        CacheKeyError: If a stable key cannot be generated for the given
            inputs.
    """
    if not namespace.strip():
        raise CacheKeyError("namespace must be a non-empty string.")

    try:
        canonical_payload = {
            "args": [_stable_value_repr(arg) for arg in args],
            "kwargs": {
                key: _stable_value_repr(value)
                for key, value in sorted(kwargs.items())
            },
        }
        serialized = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":"), default=str
        )
    except (TypeError, ValueError) as exc:
        raise CacheKeyError(
            f"Failed to generate a stable cache key for namespace "
            f"'{namespace}': {exc}"
        ) from exc

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _namespace_of(key: str) -> str:
    """
    Extract the namespace portion of a cache key.

    Args:
        key: A cache key, typically produced by ``make_cache_key``.

    Returns:
        The substring before the first colon, or the entire key if no
        colon is present.
    """
    return key.split(":", 1)[0] if ":" in key else key


# ---------------------------------------------------------------------------
# Core cache implementation
# ---------------------------------------------------------------------------


class LRUTTLCache:
    """
    A thread-safe, in-memory cache combining bounded-size LRU eviction
    with optional per-entry TTL expiration.

    All public methods are safe to call concurrently from multiple
    threads; a single re-entrant lock guards every mutation and
    read of internal state.

    Attributes:
        config: The ``CacheConfig`` governing this cache's size and
            default expiration behavior.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """
        Initialize an empty cache.

        Args:
            config: The cache configuration to use. If omitted, a
                default ``CacheConfig`` (max_size=2048, no default TTL)
                is used.
        """
        self.config = config or CacheConfig()
        self._lock = threading.RLock()
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0
        self._total_expirations = 0
        self._namespace_counters: dict[str, dict[str, int]] = {}

    def _counters_for(self, namespace: str) -> dict[str, int]:
        """
        Return (creating if necessary) the mutable statistics counters
        for a namespace. Must be called while holding ``self._lock``.

        Args:
            namespace: The namespace to fetch counters for.

        Returns:
            A mutable dict with keys "hits", "misses", "evictions", and
            "expirations".
        """
        return self._namespace_counters.setdefault(
            namespace, {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}
        )

    def _record_hit(self, key: str) -> None:
        """Record a cache hit for statistics purposes. Caller must hold
        ``self._lock``."""
        self._total_hits += 1
        if self.config.track_namespace_stats:
            self._counters_for(_namespace_of(key))["hits"] += 1

    def _record_miss(self, key: str) -> None:
        """Record a cache miss for statistics purposes. Caller must hold
        ``self._lock``."""
        self._total_misses += 1
        if self.config.track_namespace_stats:
            self._counters_for(_namespace_of(key))["misses"] += 1

    def _record_eviction(self, key: str) -> None:
        """Record an LRU eviction for statistics purposes. Caller must
        hold ``self._lock``."""
        self._total_evictions += 1
        if self.config.track_namespace_stats:
            self._counters_for(_namespace_of(key))["evictions"] += 1

    def _record_expiration(self, key: str) -> None:
        """Record a TTL expiration for statistics purposes. Caller must
        hold ``self._lock``."""
        self._total_expirations += 1
        if self.config.track_namespace_stats:
            self._counters_for(_namespace_of(key))["expirations"] += 1

    def try_get(self, key: str) -> tuple[bool, Any]:
        """
        Attempt to retrieve a value from the cache, distinguishing a
        genuine cache hit (even one whose value is ``None``) from a
        miss.

        On a hit, the entry is marked as most-recently-used. On an
        expired entry, the entry is removed and treated as a miss.

        Args:
            key: The cache key to look up.

        Returns:
            A tuple ``(found, value)``. If ``found`` is False, ``value``
            is always ``None`` and should be ignored.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._record_miss(key)
                return False, None

            now = datetime.now(timezone.utc)
            if entry.is_expired(now):
                del self._store[key]
                self._record_expiration(key)
                self._record_miss(key)
                logger.debug("Cache entry expired for key='%s'.", key)
                return False, None

            self._store.move_to_end(key)
            entry.last_accessed_at = now
            entry.hit_count += 1
            self._record_hit(key)
            return True, entry.value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the cache, or a default if absent or
        expired.

        Args:
            key: The cache key to look up.
            default: The value to return on a cache miss.

        Returns:
            The cached value, or ``default`` if the key is not present
            or has expired.
        """
        found, value = self.try_get(key)
        return value if found else default

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Store a value in the cache, evicting the least-recently-used
        entry if the cache is at capacity.

        Args:
            key: The cache key under which to store the value.
            value: The value to cache.
            ttl: Time-to-live, in seconds, for this specific entry. If
                omitted, ``self.config.default_ttl_seconds`` is used
                (which may itself be ``None``, meaning no expiration).

        Raises:
            ValueError: If ``ttl`` is provided and is not positive.
        """
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive when provided.")

        effective_ttl = ttl if ttl is not None else self.config.default_ttl_seconds
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=effective_ttl) if effective_ttl else None

        with self._lock:
            if key in self._store:
                # Remove first so re-insertion places it at the
                # most-recently-used end.
                del self._store[key]

            self._store[key] = CacheEntry(
                value=value,
                created_at=now,
                expires_at=expires_at,
                last_accessed_at=now,
            )
            self._evict_if_over_capacity()

        logger.debug(
            "Cached value for key='%s' (ttl=%s).",
            key,
            f"{effective_ttl}s" if effective_ttl else "none",
        )

    def _evict_if_over_capacity(self) -> None:
        """
        Evict least-recently-used entries until the cache is within its
        configured maximum size. Caller must hold ``self._lock``.
        """
        while len(self._store) > self.config.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            self._record_eviction(evicted_key)
            logger.debug("Evicted LRU cache entry for key='%s'.", evicted_key)

    def contains(self, key: str) -> bool:
        """
        Check whether a non-expired entry exists for the given key,
        without affecting LRU order or statistics.

        Args:
            key: The cache key to check.

        Returns:
            True if a live (non-expired) entry exists, False otherwise.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            return not entry.is_expired()

    def invalidate(self, key: str) -> bool:
        """
        Remove a single entry from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if an entry was present and removed, False otherwise.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug("Invalidated cache entry for key='%s'.", key)
                return True
            return False

    def invalidate_namespace(self, namespace: str) -> int:
        """
        Remove all entries belonging to a given namespace.

        Args:
            namespace: The namespace whose entries should be removed
                (typically a ``CacheNamespace`` value).

        Returns:
            The number of entries removed.
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._store if _namespace_of(key) == namespace
            ]
            for key in keys_to_remove:
                del self._store[key]

        if keys_to_remove:
            logger.info(
                "Invalidated %d cache entr%s in namespace '%s'.",
                len(keys_to_remove),
                "y" if len(keys_to_remove) == 1 else "ies",
                namespace,
            )
        return len(keys_to_remove)

    def clear(self, namespace: str | None = None) -> int:
        """
        Remove all entries from the cache, or all entries within a
        specific namespace.

        Args:
            namespace: If provided, only entries in this namespace are
                removed. If omitted, the entire cache is cleared.

        Returns:
            The number of entries removed.
        """
        if namespace is not None:
            return self.invalidate_namespace(namespace)

        with self._lock:
            removed_count = len(self._store)
            self._store.clear()

        logger.info("Cleared entire cache (%d entr%s removed).", removed_count, "y" if removed_count == 1 else "ies")
        return removed_count

    def prune_expired(self) -> int:
        """
        Proactively remove all currently expired entries, regardless of
        whether they have been accessed.

        This is useful for periodic background maintenance in
        long-running processes, since expired entries are otherwise only
        cleaned up lazily on access via ``get``/``try_get``.

        Returns:
            The number of expired entries removed.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            expired_keys = [
                key for key, entry in self._store.items() if entry.is_expired(now)
            ]
            for key in expired_keys:
                del self._store[key]
                self._record_expiration(key)

        if expired_keys:
            logger.info("Pruned %d expired cache entr%s.", len(expired_keys), "y" if len(expired_keys) == 1 else "ies")
        return len(expired_keys)

    def keys(self, namespace: str | None = None) -> list[str]:
        """
        List the cache keys currently stored, optionally filtered to a
        single namespace.

        Args:
            namespace: If provided, only keys in this namespace are
                returned.

        Returns:
            A list of cache keys, in least-recently-used to
            most-recently-used order.
        """
        with self._lock:
            if namespace is None:
                return list(self._store.keys())
            return [key for key in self._store if _namespace_of(key) == namespace]

    @property
    def size(self) -> int:
        """
        Return the current number of entries stored in the cache.

        Returns:
            The number of entries currently in the cache.
        """
        with self._lock:
            return len(self._store)

    def stats(self) -> CacheStatistics:
        """
        Compute a snapshot of the cache's current statistics.

        Returns:
            A populated ``CacheStatistics`` instance.
        """
        with self._lock:
            namespace_stats: dict[str, NamespaceStatistics] = {}
            if self.config.track_namespace_stats:
                entry_counts: dict[str, int] = {}
                for key in self._store:
                    ns = _namespace_of(key)
                    entry_counts[ns] = entry_counts.get(ns, 0) + 1

                all_namespaces = set(self._namespace_counters) | set(entry_counts)
                for namespace in all_namespaces:
                    counters = self._namespace_counters.get(
                        namespace, {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}
                    )
                    namespace_stats[namespace] = NamespaceStatistics(
                        namespace=namespace,
                        hits=counters["hits"],
                        misses=counters["misses"],
                        evictions=counters["evictions"],
                        expirations=counters["expirations"],
                        entry_count=entry_counts.get(namespace, 0),
                    )

            return CacheStatistics(
                total_hits=self._total_hits,
                total_misses=self._total_misses,
                total_evictions=self._total_evictions,
                total_expirations=self._total_expirations,
                current_entry_count=len(self._store),
                max_size=self.config.max_size,
                namespace_stats=namespace_stats,
            )

    def export_snapshot(
        self, serializer: Callable[[Any], Any] | None = None
    ) -> dict[str, Any]:
        """
        Export the current cache contents as a JSON-serializable
        snapshot, useful for diagnostics, logging, or transferring
        cached results across process boundaries.

        Args:
            serializer: Optional custom serializer applied to each
                cached value. If omitted, a default serializer is used
                that converts dataclass instances (via
                ``dataclasses.asdict``) and falls back to ``str()`` for
                any other non-JSON-native type.

        Returns:
            A dict mapping each cache key to a dict with ``value``,
            ``created_at``, ``expires_at``, and ``hit_count`` fields.

        Raises:
            CacheSerializationError: If a value cannot be serialized by
                the given or default serializer.
        """
        effective_serializer = serializer or _default_snapshot_serializer

        with self._lock:
            snapshot: dict[str, Any] = {}
            for key, entry in self._store.items():
                try:
                    serialized_value = effective_serializer(entry.value)
                except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
                    raise CacheSerializationError(
                        f"Failed to serialize cached value for key='{key}': {exc}"
                    ) from exc

                snapshot[key] = {
                    "value": serialized_value,
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": (
                        entry.expires_at.isoformat() if entry.expires_at else None
                    ),
                    "hit_count": entry.hit_count,
                }
            return snapshot

    def to_json(self, serializer: Callable[[Any], Any] | None = None) -> str:
        """
        Export the current cache contents as a JSON string.

        Args:
            serializer: Optional custom serializer, passed through to
                ``export_snapshot``.

        Returns:
            A JSON-formatted string representing the cache's contents.

        Raises:
            CacheSerializationError: If a value cannot be serialized.
        """
        try:
            return json.dumps(self.export_snapshot(serializer), default=str, indent=2)
        except CacheSerializationError:
            raise
        except (TypeError, ValueError) as exc:
            raise CacheSerializationError(
                f"Failed to serialize cache snapshot to JSON: {exc}"
            ) from exc

    def __len__(self) -> int:
        """Return the current number of entries stored in the cache."""
        return self.size

    def __contains__(self, key: str) -> bool:
        """Support ``key in cache`` syntax; equivalent to ``contains``."""
        return self.contains(key)


def _default_snapshot_serializer(value: Any) -> Any:
    """
    Default serializer used by ``LRUTTLCache.export_snapshot`` when no
    custom serializer is supplied.

    Converts dataclass instances (including ``drug_discovery.models``
    types) into plain dicts via ``dataclasses.asdict``, recursively
    handling ``Enum`` members and nested dataclasses. Values that are
    already JSON-native are passed through unchanged; anything else
    falls back to ``str()``.

    Args:
        value: The cached value to serialize.

    Returns:
        A JSON-serializable representation of ``value``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(
            value,
            dict_factory=lambda items: {
                key: (item.value if isinstance(item, Enum) else item)
                for key, item in items
            },
        )
    if isinstance(value, dict):
        return {str(key): _default_snapshot_serializer(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_default_snapshot_serializer(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# Default (module-level) cache singleton
# ---------------------------------------------------------------------------

_default_cache_lock = threading.Lock()
_default_cache: LRUTTLCache | None = None


def get_default_cache() -> LRUTTLCache:
    """
    Return the process-wide default cache instance, creating it on first
    use with a standard ``CacheConfig``.

    This is the cache instance used by ``cached`` (and the per-module
    convenience decorators) when no explicit cache is supplied, so that
    all Drug Discovery modules share a single cache and its statistics
    by default.

    Returns:
        The default ``LRUTTLCache`` instance.
    """
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                _default_cache = LRUTTLCache()
                logger.info("Initialized default Drug Discovery cache.")
    return _default_cache


def configure_default_cache(config: CacheConfig) -> LRUTTLCache:
    """
    Replace the process-wide default cache with a newly configured
    instance.

    Any entries in the previous default cache are discarded. This is
    typically called once, at application startup, to size the cache
    appropriately for the deployment environment.

    Args:
        config: The configuration to use for the new default cache.

    Returns:
        The newly created default ``LRUTTLCache`` instance.
    """
    global _default_cache
    with _default_cache_lock:
        _default_cache = LRUTTLCache(config)
        logger.info(
            "Reconfigured default Drug Discovery cache (max_size=%d, "
            "default_ttl_seconds=%s).",
            config.max_size,
            config.default_ttl_seconds,
        )
        return _default_cache


def reset_default_cache() -> None:
    """
    Clear the process-wide default cache without changing its
    configuration.

    Useful in tests or long-running services that need to force a full
    cache reset without altering size/TTL configuration.
    """
    with _default_cache_lock:
        if _default_cache is not None:
            removed = _default_cache.clear()
            logger.info("Reset default Drug Discovery cache (%d entries removed).", removed)


# ---------------------------------------------------------------------------
# Caching decorator
# ---------------------------------------------------------------------------


def cached(
    namespace: str,
    ttl: float | None = None,
    cache: LRUTTLCache | None = None,
    key_builder: Callable[..., str] | None = None,
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Decorator that transparently caches a function's return value,
    keyed by its namespace and arguments.

    Intended for wrapping expensive, deterministic Drug Discovery
    computations (e.g., ``calculate_molecular_properties``,
    ``find_similar_compounds``, ``predict_targets``,
    ``predict_toxicity``) so repeated calls with the same effective
    arguments are served from the cache instead of being recomputed.

    Args:
        namespace: The cache namespace to use for this function's
            entries (typically a ``CacheNamespace`` value).
        ttl: Time-to-live, in seconds, for entries produced by this
            function. If omitted, the cache's configured default TTL is
            used.
        cache: The ``LRUTTLCache`` instance to use. If omitted, the
            process-wide default cache (``get_default_cache``) is used.
        key_builder: Optional custom function that receives the same
            positional and keyword arguments as the wrapped function and
            returns a cache key string. If omitted, a key is generated
            automatically via ``make_cache_key`` from the namespace, the
            function's qualified name, and all arguments.

    Returns:
        A decorator that wraps a function with caching behavior. The
        wrapped function gains a ``cache_clear()`` method (invalidating
        only this function's namespace) and a ``cache_stats()`` method
        (returning the underlying cache's full ``CacheStatistics``).

    Raises:
        CacheKeyError: If a stable cache key cannot be generated for a
            given call's arguments (propagated from ``make_cache_key``).
    """

    def decorator(func: Callable[..., _R]) -> Callable[..., _R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _R:
            active_cache = cache or get_default_cache()

            if key_builder is not None:
                key = key_builder(*args, **kwargs)
            else:
                key = make_cache_key(namespace, func.__qualname__, *args, **kwargs)

            found, cached_value = active_cache.try_get(key)
            if found:
                logger.debug(
                    "Cache hit for '%s' (namespace='%s').",
                    func.__qualname__,
                    namespace,
                )
                return cached_value  # type: ignore[return-value]

            logger.debug(
                "Cache miss for '%s' (namespace='%s'); computing result.",
                func.__qualname__,
                namespace,
            )
            result = func(*args, **kwargs)
            active_cache.set(key, result, ttl=ttl)
            return result

        def cache_clear() -> int:
            """Invalidate all cache entries in this function's namespace."""
            active_cache = cache or get_default_cache()
            return active_cache.invalidate_namespace(namespace)

        def cache_stats() -> CacheStatistics:
            """Return the underlying cache's current statistics."""
            active_cache = cache or get_default_cache()
            return active_cache.stats()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_stats = cache_stats  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Per-module convenience decorators
# ---------------------------------------------------------------------------


def cache_molecular_properties(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``molecular_properties`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.MOLECULAR_PROPERTIES``.
    """
    return cached(CacheNamespace.MOLECULAR_PROPERTIES.value, ttl=ttl, cache=cache)


def cache_compound_similarity(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``compound_similarity`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.COMPOUND_SIMILARITY``.
    """
    return cached(CacheNamespace.COMPOUND_SIMILARITY.value, ttl=ttl, cache=cache)


def cache_target_prediction(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``target_prediction`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.TARGET_PREDICTION``.
    """
    return cached(CacheNamespace.TARGET_PREDICTION.value, ttl=ttl, cache=cache)


def cache_toxicity_prediction(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``toxicity_prediction`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.TOXICITY_PREDICTION``.
    """
    return cached(CacheNamespace.TOXICITY_PREDICTION.value, ttl=ttl, cache=cache)


def cache_drug_repurposing(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``drug_repurposing`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.DRUG_REPURPOSING``.
    """
    return cached(CacheNamespace.DRUG_REPURPOSING.value, ttl=ttl, cache=cache)


def cache_candidate_ranking(
    ttl: float | None = None, cache: LRUTTLCache | None = None
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    """
    Convenience decorator for caching ``candidate_ranking`` results.

    Args:
        ttl: Optional time-to-live override, in seconds.
        cache: Optional explicit cache instance to use.

    Returns:
        A caching decorator scoped to
        ``CacheNamespace.CANDIDATE_RANKING``.
    """
    return cached(CacheNamespace.CANDIDATE_RANKING.value, ttl=ttl, cache=cache)