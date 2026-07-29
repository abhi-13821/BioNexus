"""
Data models and schemas for the BioNexus Embeddings module.

This module defines the core, dependency-light data structures shared by
``embedding_generator.py``, ``embedding_store.py``, and
``embedding_search.py``. Centralizing these models here ensures a single
source of truth for the shape of embedding data flowing through the
system, and keeps the module decoupled from any specific storage or
inference backend (Dependency Inversion Principle).

All models use the standard library ``dataclasses`` module (no external
schema library dependency such as Pydantic is assumed here, since the
existing BioNexus modules were not confirmed to use one). Validation is
performed explicitly in ``__post_init__`` hooks to guarantee that invalid
data cannot be constructed, which keeps downstream code free of defensive
checks.

Classes
-------
SimilarityMetric
    Enumeration of supported vector similarity metrics.
EmbeddingModelConfig
    Configuration describing which embedding model to use and how.
EmbeddingMetadata
    Metadata describing the provenance of a single embedding (paper it
    came from, section of text, source module, etc.).
EmbeddingRecord
    A single embedding vector bound to its metadata and a unique ID.
SimilaritySearchResult
    A single scored result returned from a semantic similarity search.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SimilarityMetric(str, Enum):
    """
    Supported vector similarity metrics for semantic search.

    Attributes
    ----------
    COSINE:
        Cosine similarity. Recommended default for sentence-transformer
        embeddings, which are typically normalized or near-normalized.
    DOT_PRODUCT:
        Raw dot product similarity. Equivalent to cosine similarity when
        vectors are L2-normalized, but cheaper to compute.
    EUCLIDEAN:
        Negative L2 (Euclidean) distance, used when absolute vector
        magnitude carries meaningful information.
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """
    Configuration describing an embedding model and how it should be run.

    This object is intentionally backend-agnostic: it describes *what*
    model to use and *how* to run it, without coupling callers to any
    specific inference library. ``embedding_generator.py`` consumes this
    configuration to instantiate the actual model.

    Attributes
    ----------
    model_name:
        Identifier of the embedding model, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``. Expected to be a
        Hugging Face Hub model identifier compatible with the
        ``sentence-transformers`` library.
    embedding_dimension:
        Dimensionality of the vectors produced by ``model_name``. Stored
        explicitly (rather than inferred at runtime only) so that
        ``embedding_store.py`` can validate incoming vectors and
        pre-allocate storage/index structures without loading the model.
    max_sequence_length:
        Maximum number of tokens the model will consider per input. Text
        longer than this will be truncated by the underlying model/
        tokenizer. Used by callers to decide on chunking strategy.
    normalize_embeddings:
        Whether output vectors should be L2-normalized. Normalized
        vectors make cosine similarity and dot product equivalent, which
        simplifies downstream FAISS index selection.
    batch_size:
        Default batch size to use when encoding multiple texts at once.
    device:
        Compute device hint (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
        A value of ``"auto"`` defers device selection to the generator
        implementation.

    Raises
    ------
    ValueError
        If ``embedding_dimension``, ``max_sequence_length``, or
        ``batch_size`` are not positive integers, or if ``model_name`` is
        empty.
    """

    model_name: str
    embedding_dimension: int
    max_sequence_length: int = 256
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got "
                f"{self.embedding_dimension}."
            )
        if self.max_sequence_length <= 0:
            raise ValueError(
                f"max_sequence_length must be positive, got "
                f"{self.max_sequence_length}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}."
            )


@dataclass(frozen=True)
class EmbeddingMetadata:
    """
    Provenance and contextual metadata for a single embedding.

    This links an embedding vector back to the biomedical literature it
    was derived from, allowing search results to be traced back to a
    source paper and text span, and enabling cross-referencing with the
    Literature Search and Knowledge Graph modules.

    Attributes
    ----------
    source_id:
        Identifier of the originating document, expected to match the
        identifier used by the Literature Search module (e.g. a DOI,
        PubMed ID, or internal paper ID).
    source_type:
        Type of the source text, e.g. ``"title"``, ``"abstract"``,
        ``"full_text_chunk"``. Allows filtering/search to be scoped to a
        particular kind of content.
    title:
        Title of the source paper, stored redundantly for fast display
        in search results without a secondary lookup.
    chunk_index:
        Index of this text chunk within its source document, for content
        that has been split into multiple embeddings (e.g. full-text
        segments). ``0`` for single-chunk sources such as titles or
        abstracts.
    text:
        The exact text that was embedded. Retained so that search
        results can display the matched content directly.
    extra:
        Free-form dictionary for additional metadata (authors, journal,
        year, DOI, citation count, knowledge-graph entity IDs, etc.)
        without requiring schema changes here. Consumers should treat
        keys defensively (use ``.get``) since this is not a fixed schema.

    Raises
    ------
    ValueError
        If ``source_id`` or ``text`` are empty, or ``chunk_index`` is
        negative.
    """

    source_id: str
    source_type: str
    title: str
    text: str
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string.")
        if not self.text or not self.text.strip():
            raise ValueError("text must be a non-empty string.")
        if self.chunk_index < 0:
            raise ValueError(
                f"chunk_index must be >= 0, got {self.chunk_index}."
            )


@dataclass(frozen=True)
class EmbeddingRecord:
    """
    A single embedding vector bound to its metadata and a unique ID.

    This is the primary unit of storage handled by
    ``embedding_store.py`` and the primary unit of comparison used by
    ``embedding_search.py``.

    Attributes
    ----------
    embedding_id:
        Globally unique identifier for this record. Auto-generated via
        UUID4 if not supplied, so callers never need to manage ID
        uniqueness manually.
    vector:
        The dense embedding vector, represented as a list of floats for
        storage/serialization portability (e.g. JSON, database columns).
        Numeric backends (e.g. NumPy, FAISS) convert this at their
        boundary.
    metadata:
        The :class:`EmbeddingMetadata` describing this vector's
        provenance.
    model_name:
        Name of the embedding model used to produce ``vector``. Stored
        per-record (not just per-store) so a store can safely contain
        embeddings from more than one model generation, and mixed-model
        comparisons can be detected and rejected at search time.
    created_at:
        UTC timestamp of when this record was created.

    Raises
    ------
    ValueError
        If ``vector`` is empty or ``model_name`` is empty.
    """

    metadata: EmbeddingMetadata
    vector: list[float]
    model_name: str
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("vector must not be empty.")
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

    @property
    def dimension(self) -> int:
        """Return the dimensionality of the stored embedding vector."""
        return len(self.vector)


@dataclass(frozen=True)
class SimilaritySearchResult:
    """
    A single scored result returned from a semantic similarity search.

    Attributes
    ----------
    record:
        The matched :class:`EmbeddingRecord`.
    score:
        Similarity score with respect to the query vector. Higher is
        always more similar, regardless of the underlying
        :class:`SimilarityMetric` used (implementations are responsible
        for normalizing distance-based metrics into a "higher is better"
        score before constructing this object).
    rank:
        1-based rank of this result within its result set (``1`` is the
        most similar).

    Raises
    ------
    ValueError
        If ``rank`` is less than 1.
    """

    record: EmbeddingRecord
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}.")


__all__: list[str] = [
    "SimilarityMetric",
    "EmbeddingModelConfig",
    "EmbeddingMetadata",
    "EmbeddingRecord",
    "SimilaritySearchResult",
]