"""
High-level orchestration for the BioNexus Embeddings module.

:class:`EmbeddingManager` is the single entry point most callers (frontend
pages, ingestion pipelines from Literature Search, and future
Retrieval-Augmented Generation / Multi-Agent components) should use. It
wires together:

    - :class:`~embeddings.embedding_generator.BaseEmbeddingGenerator`
      (text -> vector),
    - :class:`~embeddings.embedding_store.BaseEmbeddingStore`
      (persistence),
    - :class:`~embeddings.cache.BaseEmbeddingCache`
      (avoiding redundant re-embedding of identical text), and
    - :class:`~embeddings.embedding_search.BaseEmbeddingSearchEngine`
      (semantic similarity search),

into a single, cohesive API, so that callers do not need to understand
or coordinate these four components individually.

Classes
-------
EmbeddingManager
    High-level facade over generation, storage, caching, and search.

Exceptions
----------
EmbeddingManagerError
    Raised when a manager-level operation fails.
"""

from __future__ import annotations

import logging

from embeddings.cache import BaseEmbeddingCache, CacheStats, LRUEmbeddingCache
from embeddings.embedding_generator import BaseEmbeddingGenerator
from embeddings.embedding_search import BaseEmbeddingSearchEngine, InMemoryEmbeddingSearchEngine
from embeddings.embedding_store import BaseEmbeddingStore, EmbeddingNotFoundError
from embeddings.models import EmbeddingMetadata, EmbeddingRecord, SimilaritySearchResult
from embeddings.utils import compute_cache_key

logger = logging.getLogger(__name__)


class EmbeddingManagerError(RuntimeError):
    """Raised when a high-level embedding management operation fails."""


class EmbeddingManager:
    """
    High-level facade coordinating embedding generation, storage, caching,
    and semantic search.

    This is the primary class that other BioNexus modules (Literature
    Search ingestion, frontend semantic search pages, and future RAG /
    Multi-Agent components) should depend on, rather than constructing
    and coordinating :class:`~embeddings.embedding_generator.BaseEmbeddingGenerator`,
    :class:`~embeddings.embedding_store.BaseEmbeddingStore`, and
    :class:`~embeddings.embedding_search.BaseEmbeddingSearchEngine`
    individually.

    Parameters
    ----------
    generator:
        The embedding generator used to convert text into vectors.
    store:
        The embedding store used to persist and retrieve records.
    cache:
        Optional cache used to avoid re-embedding identical
        ``(model_name, text)`` pairs. If ``None``, a default
        :class:`~embeddings.cache.LRUEmbeddingCache` with
        ``max_size=10_000`` is created automatically.
    search_engine:
        Optional search engine used for semantic similarity search. If
        ``None``, a default
        :class:`~embeddings.embedding_search.InMemoryEmbeddingSearchEngine`
        is constructed from ``generator`` and ``store``.

    Examples
    --------
    >>> manager = EmbeddingManager(generator, store)  # doctest: +SKIP
    >>> record = manager.add_document(metadata)  # doctest: +SKIP
    >>> results = manager.search("EGFR mutation lung cancer")  # doctest: +SKIP
    """

    def __init__(
        self,
        generator: BaseEmbeddingGenerator,
        store: BaseEmbeddingStore,
        cache: BaseEmbeddingCache | None = None,
        search_engine: BaseEmbeddingSearchEngine | None = None,
    ) -> None:
        self._generator = generator
        self._store = store
        self._cache: BaseEmbeddingCache = (
            cache if cache is not None else LRUEmbeddingCache(max_size=10_000)
        )
        self._search_engine: BaseEmbeddingSearchEngine = (
            search_engine
            if search_engine is not None
            else InMemoryEmbeddingSearchEngine(generator, store)
        )

        logger.debug(
            "EmbeddingManager initialized with model='%s', cache=%s, "
            "search_engine=%s.",
            self._generator.config.model_name,
            type(self._cache).__name__,
            type(self._search_engine).__name__,
        )

    @property
    def generator(self) -> BaseEmbeddingGenerator:
        """Return the embedding generator used by this manager."""
        return self._generator

    @property
    def store(self) -> BaseEmbeddingStore:
        """Return the embedding store used by this manager."""
        return self._store

    @property
    def cache(self) -> BaseEmbeddingCache:
        """Return the embedding cache used by this manager."""
        return self._cache

    @property
    def search_engine(self) -> BaseEmbeddingSearchEngine:
        """Return the semantic search engine used by this manager."""
        return self._search_engine

    def embed_text(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate an embedding vector for raw text, using the cache when possible.

        Parameters
        ----------
        text:
            The text to embed. Must be non-empty.
        use_cache:
            Whether to check and populate the cache for this call. Set to
            ``False`` to force fresh generation (e.g. for cache
            invalidation testing or benchmarking).

        Returns
        -------
        list[float]
            The embedding vector.

        Raises
        ------
        ValueError
            If ``text`` is empty or whitespace-only.
        EmbeddingManagerError
            If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        model_name = self._generator.config.model_name
        cache_key = compute_cache_key(model_name, text) if use_cache else None

        if use_cache and cache_key is not None:
            cached_vector = self._cache.get(cache_key)
            if cached_vector is not None:
                logger.debug("Cache hit for text embedding (model='%s').", model_name)
                return cached_vector

        try:
            vector = self._generator.generate(text)
        except Exception as exc:
            logger.exception("EmbeddingManager failed to generate embedding.")
            raise EmbeddingManagerError(
                f"Failed to generate embedding: {exc}"
            ) from exc

        if use_cache and cache_key is not None:
            self._cache.put(cache_key, vector)

        return vector

    def add_document(
        self,
        metadata: EmbeddingMetadata,
        use_cache: bool = True,
    ) -> EmbeddingRecord:
        """
        Embed and store a single document's text, returning the stored record.

        Parameters
        ----------
        metadata:
            Metadata describing the document text to embed and store.
        use_cache:
            Whether to use the cache when generating the embedding.

        Returns
        -------
        EmbeddingRecord
            The record that was persisted to the store.

        Raises
        ------
        EmbeddingManagerError
            If embedding generation or storage fails.
        """
        vector = self.embed_text(metadata.text, use_cache=use_cache)
        record = EmbeddingRecord(
            metadata=metadata,
            vector=vector,
            model_name=self._generator.config.model_name,
        )

        try:
            self._store.add(record)
        except Exception as exc:
            logger.exception(
                "EmbeddingManager failed to store record for source_id='%s'.",
                metadata.source_id,
            )
            raise EmbeddingManagerError(
                f"Failed to store embedding record: {exc}"
            ) from exc

        logger.debug(
            "Stored embedding record '%s' for source_id='%s'.",
            record.embedding_id,
            metadata.source_id,
        )
        return record

    def add_documents(
        self,
        metadata_list: list[EmbeddingMetadata],
        use_cache: bool = True,
    ) -> list[EmbeddingRecord]:
        """
        Embed and store multiple documents in one call.

        Parameters
        ----------
        metadata_list:
            The list of document metadata to embed and store. Must not
            be empty.
        use_cache:
            Whether to use the cache when generating embeddings.

        Returns
        -------
        list[EmbeddingRecord]
            The records that were persisted, in the same order as
            ``metadata_list``.

        Raises
        ------
        ValueError
            If ``metadata_list`` is empty.
        EmbeddingManagerError
            If embedding generation or storage fails for any document.
        """
        if not metadata_list:
            raise ValueError("metadata_list must not be empty.")

        records = [
            EmbeddingRecord(
                metadata=metadata,
                vector=self.embed_text(metadata.text, use_cache=use_cache),
                model_name=self._generator.config.model_name,
            )
            for metadata in metadata_list
        ]

        try:
            self._store.add_batch(records)
        except Exception as exc:
            logger.exception(
                "EmbeddingManager failed to store a batch of %d record(s).",
                len(records),
            )
            raise EmbeddingManagerError(
                f"Failed to store embedding records: {exc}"
            ) from exc

        logger.debug("Stored %d embedding record(s) in batch.", len(records))
        return records

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[SimilaritySearchResult]:
        """
        Perform a semantic similarity search for a free-text query.

        Parameters
        ----------
        query:
            The free-text query.
        top_k:
            Maximum number of results to return.
        source_type:
            If provided, restrict results to this source type.

        Returns
        -------
        list[SimilaritySearchResult]
            Ranked search results, most similar first.

        Raises
        ------
        ValueError
            If ``query`` is empty or ``top_k`` is not positive.
        EmbeddingManagerError
            If the search fails.
        """
        try:
            return self._search_engine.search(query, top_k=top_k, source_type=source_type)
        except ValueError:
            raise
        except Exception as exc:
            logger.exception("EmbeddingManager search failed for query %r.", query)
            raise EmbeddingManagerError(f"Search failed: {exc}") from exc

    def find_similar(self, embedding_id: str, top_k: int = 10) -> list[SimilaritySearchResult]:
        """
        Find records similar to an already-stored record.

        Parameters
        ----------
        embedding_id:
            The ID of the stored record to use as the query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[SimilaritySearchResult]
            Ranked results, most similar first, excluding the queried
            record itself.

        Raises
        ------
        ValueError
            If ``top_k`` is not positive.
        EmbeddingManagerError
            If ``embedding_id`` does not exist or the search fails.
        """
        try:
            return self._search_engine.find_similar(embedding_id, top_k=top_k)
        except ValueError:
            raise
        except Exception as exc:
            logger.exception(
                "EmbeddingManager find_similar failed for embedding_id='%s'.",
                embedding_id,
            )
            raise EmbeddingManagerError(f"find_similar failed: {exc}") from exc

    def get(self, embedding_id: str) -> EmbeddingRecord:
        """
        Retrieve a single stored record by ID.

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
        EmbeddingManagerError
            If no record with the given ID exists.
        """
        try:
            return self._store.get(embedding_id)
        except EmbeddingNotFoundError as exc:
            raise EmbeddingManagerError(str(exc)) from exc

    def get_by_source(self, source_id: str) -> list[EmbeddingRecord]:
        """
        Retrieve all records derived from a given source document.

        Parameters
        ----------
        source_id:
            The source document identifier.

        Returns
        -------
        list[EmbeddingRecord]
            All matching records, ordered by chunk index.
        """
        return self._store.get_by_source(source_id)

    def delete(self, embedding_id: str) -> None:
        """
        Delete a stored record by ID.

        Parameters
        ----------
        embedding_id:
            The unique ID of the record to delete.

        Raises
        ------
        EmbeddingManagerError
            If no record with the given ID exists.
        """
        try:
            self._store.delete(embedding_id)
        except EmbeddingNotFoundError as exc:
            raise EmbeddingManagerError(str(exc)) from exc

    def count(self) -> int:
        """
        Return the total number of stored embedding records.

        Returns
        -------
        int
            The number of records in the store.
        """
        return self._store.count()

    def cache_stats(self) -> CacheStats:
        """
        Return current cache performance statistics.

        Returns
        -------
        CacheStats
            A snapshot of cache hits, misses, evictions, expirations, and
            size.
        """
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache without affecting stored records."""
        self._cache.clear()


__all__: list[str] = [
    "EmbeddingManagerError",
    "EmbeddingManager",
]