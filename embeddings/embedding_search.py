"""
Semantic similarity search for the BioNexus Embeddings module.

This module ties together :class:`~embeddings.embedding_generator.BaseEmbeddingGenerator`
(to embed a free-text query) and :class:`~embeddings.embedding_store.BaseEmbeddingStore`
(to supply candidate vectors) to perform semantic similarity search over
biomedical literature embeddings.

The current implementation performs an in-memory brute-force similarity
scan over all records returned by the store. This is correct and fast
enough for research-scale datasets (the target for this phase) and is
deliberately isolated behind the :class:`BaseEmbeddingSearchEngine`
interface so it can later be replaced by a FAISS-backed engine (using an
approximate nearest-neighbor index) without changing any calling code in
the frontend or a future RAG pipeline.

Classes
-------
BaseEmbeddingSearchEngine
    Abstract interface for semantic similarity search.
InMemoryEmbeddingSearchEngine
    Brute-force cosine/dot-product/Euclidean similarity search over all
    records in a :class:`BaseEmbeddingStore`.

Exceptions
----------
EmbeddingSearchError
    Raised when a search operation fails.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from embeddings.embedding_generator import BaseEmbeddingGenerator
from embeddings.embedding_store import BaseEmbeddingStore
from embeddings.models import EmbeddingRecord, SimilarityMetric, SimilaritySearchResult

logger = logging.getLogger(__name__)


class EmbeddingSearchError(RuntimeError):
    """Raised when a semantic similarity search operation fails."""


class BaseEmbeddingSearchEngine(ABC):
    """
    Abstract interface for semantic similarity search over embeddings.

    Implementations combine a query embedding step (via a
    :class:`~embeddings.embedding_generator.BaseEmbeddingGenerator`) with
    a candidate-retrieval step (via a
    :class:`~embeddings.embedding_store.BaseEmbeddingStore`) to return a
    ranked list of the most semantically similar records to a query.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[SimilaritySearchResult]:
        """
        Search for embedding records most similar to a free-text query.

        Parameters
        ----------
        query:
            The free-text query to search for (e.g. a research question
            or topic phrase). Must be non-empty.
        top_k:
            Maximum number of results to return. Must be positive.
        source_type:
            If provided, restrict the search to records whose
            ``metadata.source_type`` equals this value (e.g. only
            ``"abstract"``). If ``None``, all source types are searched.

        Returns
        -------
        list[SimilaritySearchResult]
            Results ordered from most to least similar, with ``rank``
            starting at 1. Fewer than ``top_k`` results are returned if
            fewer candidates exist.

        Raises
        ------
        ValueError
            If ``query`` is empty or ``top_k`` is not positive.
        EmbeddingSearchError
            If the search fails (e.g. query embedding generation fails).
        """
        raise NotImplementedError

    @abstractmethod
    def find_similar(
        self,
        embedding_id: str,
        top_k: int = 10,
    ) -> list[SimilaritySearchResult]:
        """
        Find records most similar to an already-stored record.

        Useful for "more like this paper" style features, without
        needing to re-embed any text.

        Parameters
        ----------
        embedding_id:
            The ID of the stored record to use as the query vector.
        top_k:
            Maximum number of results to return, excluding the queried
            record itself. Must be positive.

        Returns
        -------
        list[SimilaritySearchResult]
            Results ordered from most to least similar, with ``rank``
            starting at 1. The record matching ``embedding_id`` itself is
            excluded from the results.

        Raises
        ------
        ValueError
            If ``top_k`` is not positive.
        EmbeddingSearchError
            If ``embedding_id`` does not exist or the search fails.
        """
        raise NotImplementedError


class InMemoryEmbeddingSearchEngine(BaseEmbeddingSearchEngine):
    """
    Brute-force, in-memory semantic similarity search engine.

    On each search, this engine loads all candidate records from the
    configured :class:`~embeddings.embedding_store.BaseEmbeddingStore`
    and scores each one against the query vector using the configured
    :class:`~embeddings.models.SimilarityMetric`. This is O(n) per query
    in the number of stored records, which is appropriate for
    research-scale corpora; large-scale deployments should switch to a
    FAISS-backed :class:`BaseEmbeddingSearchEngine` implementation, which
    can be introduced later as a drop-in replacement.

    Parameters
    ----------
    generator:
        Embedding generator used to embed free-text queries. Must use
        the same model as the records in ``store`` — this is not
        enforced automatically, since the store may legitimately hold
        records from more than one model generation, but mixing models
        between query and candidates will produce meaningless scores.
    store:
        Embedding store providing candidate records to search over.
    metric:
        Similarity metric to use when scoring candidates. Defaults to
        cosine similarity, appropriate for normalized
        sentence-transformer embeddings.

    Examples
    --------
    >>> engine = InMemoryEmbeddingSearchEngine(generator, store)  # doctest: +SKIP
    >>> results = engine.search("EGFR mutation lung cancer", top_k=5)  # doctest: +SKIP
    """

    def __init__(
        self,
        generator: BaseEmbeddingGenerator,
        store: BaseEmbeddingStore,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> None:
        self._generator = generator
        self._store = store
        self._metric = metric

    @property
    def metric(self) -> SimilarityMetric:
        """Return the similarity metric used by this engine."""
        return self._metric

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[SimilaritySearchResult]:
        """
        Search for records similar to a free-text query.

        See :meth:`BaseEmbeddingSearchEngine.search` for full contract.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        try:
            query_vector = self._generator.generate(query)
        except Exception as exc:
            logger.exception("Failed to embed query for search: %r", query)
            raise EmbeddingSearchError(
                f"Failed to embed query for search: {exc}"
            ) from exc

        candidates = self._store.all()
        if source_type is not None:
            candidates = [
                record
                for record in candidates
                if record.metadata.source_type == source_type
            ]

        logger.debug(
            "Searching %d candidate record(s) for query %r (top_k=%d, "
            "source_type=%s, metric=%s).",
            len(candidates),
            query,
            top_k,
            source_type,
            self._metric.value,
        )

        return self._rank(query_vector, candidates, top_k, exclude_id=None)

    def find_similar(
        self,
        embedding_id: str,
        top_k: int = 10,
    ) -> list[SimilaritySearchResult]:
        """
        Find records similar to an already-stored record.

        See :meth:`BaseEmbeddingSearchEngine.find_similar` for full
        contract.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        try:
            source_record = self._store.get(embedding_id)
        except Exception as exc:
            logger.exception(
                "Failed to retrieve source record '%s' for find_similar.",
                embedding_id,
            )
            raise EmbeddingSearchError(
                f"Failed to retrieve record '{embedding_id}': {exc}"
            ) from exc

        candidates = self._store.all()
        logger.debug(
            "Finding records similar to '%s' among %d candidate(s) "
            "(top_k=%d, metric=%s).",
            embedding_id,
            len(candidates),
            top_k,
            self._metric.value,
        )

        return self._rank(
            source_record.vector, candidates, top_k, exclude_id=embedding_id
        )

    def _rank(
        self,
        query_vector: list[float],
        candidates: list[EmbeddingRecord],
        top_k: int,
        exclude_id: str | None,
    ) -> list[SimilaritySearchResult]:
        """
        Score and rank candidate records against a query vector.

        Parameters
        ----------
        query_vector:
            The vector to compare candidates against.
        candidates:
            The pool of candidate records to score.
        top_k:
            Maximum number of results to return.
        exclude_id:
            If provided, a record with this ``embedding_id`` is excluded
            from scoring (used by :meth:`find_similar` to avoid a record
            matching itself with a perfect score).

        Returns
        -------
        list[SimilaritySearchResult]
            Ranked results, most similar first.

        Raises
        ------
        EmbeddingSearchError
            If scoring fails (e.g. dimension mismatch between query and
            a candidate vector).
        """
        scored: list[tuple[float, EmbeddingRecord]] = []

        for record in candidates:
            if exclude_id is not None and record.embedding_id == exclude_id:
                continue
            try:
                score = self._compute_similarity(query_vector, record.vector)
            except ValueError as exc:
                logger.warning(
                    "Skipping record '%s' during scoring: %s",
                    record.embedding_id,
                    exc,
                )
                continue
            scored.append((score, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_results = scored[:top_k]

        return [
            SimilaritySearchResult(record=record, score=score, rank=rank)
            for rank, (score, record) in enumerate(top_results, start=1)
        ]

    def _compute_similarity(
        self, vector_a: list[float], vector_b: list[float]
    ) -> float:
        """
        Compute the configured similarity metric between two vectors.

        Parameters
        ----------
        vector_a:
            First vector (typically the query).
        vector_b:
            Second vector (typically a candidate).

        Returns
        -------
        float
            The similarity score. Higher always means more similar,
            regardless of metric.

        Raises
        ------
        ValueError
            If the vectors have mismatched dimensions.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError(
                f"Vector dimension mismatch: {len(vector_a)} vs "
                f"{len(vector_b)}."
            )

        if self._metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(vector_a, vector_b)
        if self._metric == SimilarityMetric.DOT_PRODUCT:
            return self._dot_product(vector_a, vector_b)
        if self._metric == SimilarityMetric.EUCLIDEAN:
            return -self._euclidean_distance(vector_a, vector_b)

        raise ValueError(f"Unsupported similarity metric: {self._metric}")

    @staticmethod
    def _dot_product(vector_a: list[float], vector_b: list[float]) -> float:
        """Compute the dot product of two equal-length vectors."""
        return sum(a * b for a, b in zip(vector_a, vector_b))

    @staticmethod
    def _euclidean_distance(
        vector_a: list[float], vector_b: list[float]
    ) -> float:
        """Compute the Euclidean (L2) distance between two equal-length vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))

    @classmethod
    def _cosine_similarity(
        cls, vector_a: list[float], vector_b: list[float]
    ) -> float:
        """
        Compute cosine similarity between two equal-length vectors.

        Returns ``0.0`` if either vector has zero magnitude, to avoid a
        division-by-zero error for degenerate (all-zero) embeddings.
        """
        dot = cls._dot_product(vector_a, vector_b)
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)


__all__: list[str] = [
    "EmbeddingSearchError",
    "BaseEmbeddingSearchEngine",
    "InMemoryEmbeddingSearchEngine",
]