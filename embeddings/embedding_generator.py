"""
Embedding generation for the BioNexus Embeddings module.

This module is responsible for converting raw biomedical text (paper
titles, abstracts, or full-text chunks) into dense vector embeddings
using a Sentence-Transformers model. It depends only on the shared
schemas defined in ``models.py`` and exposes an abstract interface
(``BaseEmbeddingGenerator``) so that ``embedding_store.py`` and
``embedding_search.py`` — and future RAG/Multi-Agent components — depend
on an abstraction rather than a concrete model implementation
(Dependency Inversion Principle).

Default Model
--------------
The default model is ``sentence-transformers/all-MiniLM-L6-v2``:
    - 384-dimensional embeddings.
    - Strong general-purpose semantic similarity performance.
    - Small (~80MB), fast on CPU, free and open-source (Apache 2.0).
    - No biomedical-specific free model (e.g. PubMedBERT-based sentence
      encoders) reliably outperforms it *and* matches its combination of
      speed, size, and ease of deployment for a research-grade platform
      at this stage. Domain-specific models can be swapped in later via
      ``EmbeddingModelConfig`` without changing any code in this module,
      since model selection is fully externalized as configuration.

Classes
-------
BaseEmbeddingGenerator
    Abstract interface for all embedding generators.
SentenceTransformerEmbeddingGenerator
    Concrete implementation backed by the ``sentence-transformers``
    library.

Exceptions
----------
EmbeddingGenerationError
    Raised when text-to-vector generation fails for any reason.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer

import logging
from abc import ABC, abstractmethod

from embeddings.models import EmbeddingMetadata, EmbeddingModelConfig, EmbeddingRecord

logger = logging.getLogger(__name__)

#: Default, free, general-purpose embedding model configuration.
#: 384-dim output, 256-token max sequence length, L2-normalized vectors.
DEFAULT_EMBEDDING_MODEL_CONFIG = EmbeddingModelConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dimension=384,
    max_sequence_length=256,
    normalize_embeddings=True,
    batch_size=32,
    device="auto",
)


class EmbeddingGenerationError(RuntimeError):
    """
    Raised when embedding generation fails.

    This wraps lower-level errors from the underlying model backend
    (e.g. model loading failures, tokenization errors, out-of-memory
    conditions) into a single, module-specific exception type so that
    callers (``embedding_store.py``, frontend modules, RAG pipelines)
    can catch one well-defined exception instead of depending on
    third-party library internals.
    """


class BaseEmbeddingGenerator(ABC):
    """
    Abstract interface for converting text into embedding vectors.

    Concrete implementations are responsible for loading a specific
    embedding model and encoding text into fixed-length vectors. This
    abstraction allows BioNexus to swap embedding backends (e.g. a
    different Sentence-Transformers model, a hosted embedding API, or a
    biomedical-specific model) without changing any code that depends on
    ``BaseEmbeddingGenerator``.
    """

    @property
    @abstractmethod
    def config(self) -> EmbeddingModelConfig:
        """Return the :class:`EmbeddingModelConfig` used by this generator."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, text: str) -> list[float]:
        """
        Generate a single embedding vector for the given text.

        Parameters
        ----------
        text:
            The raw text to embed. Must be non-empty after stripping
            whitespace.

        Returns
        -------
        list[float]
            The dense embedding vector.

        Raises
        ------
        ValueError
            If ``text`` is empty or whitespace-only.
        EmbeddingGenerationError
            If the underlying model fails to produce an embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of texts.

        Parameters
        ----------
        texts:
            A list of raw texts to embed. Must be non-empty, and no
            individual text may be empty or whitespace-only.

        Returns
        -------
        list[list[float]]
            A list of dense embedding vectors, in the same order as
            ``texts``.

        Raises
        ------
        ValueError
            If ``texts`` is empty, or any element is empty/whitespace-only.
        EmbeddingGenerationError
            If the underlying model fails to produce embeddings.
        """
        raise NotImplementedError

    def generate_record(self, metadata: EmbeddingMetadata) -> EmbeddingRecord:
        """
        Generate an :class:`EmbeddingRecord` from an :class:`EmbeddingMetadata`.

        This is a convenience method built on top of :meth:`generate`
        that binds the resulting vector to its source metadata and the
        current model name, producing a fully-formed, storable record.

        Parameters
        ----------
        metadata:
            Metadata describing the text to embed. ``metadata.text`` is
            the value that will actually be encoded.

        Returns
        -------
        EmbeddingRecord
            A record containing the generated vector and the supplied
            metadata.

        Raises
        ------
        EmbeddingGenerationError
            If the underlying model fails to produce an embedding.
        """
        vector = self.generate(metadata.text)
        return EmbeddingRecord(
            metadata=metadata,
            vector=vector,
            model_name=self.config.model_name,
        )


class SentenceTransformerEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Embedding generator backed by the ``sentence-transformers`` library.

    The underlying model is loaded lazily on first use (rather than at
    construction time) so that importing this module — or constructing
    this class — never triggers a network call or disk load by itself.
    This keeps unit tests and lightweight tooling fast, and ensures
    model loading failures surface at a predictable point (first call to
    :meth:`generate` or :meth:`generate_batch`), not at import time.

    Parameters
    ----------
    model_config:
        Configuration describing which model to load and how to run it.
        Defaults to :data:`DEFAULT_EMBEDDING_MODEL_CONFIG`
        (``sentence-transformers/all-MiniLM-L6-v2``).

    Examples
    --------
    >>> generator = SentenceTransformerEmbeddingGenerator()
    >>> vector = generator.generate("EGFR mutations drive lung cancer.")
    >>> len(vector) == generator.config.embedding_dimension
    True
    """

    def __init__(
        self,
        model_config: EmbeddingModelConfig = DEFAULT_EMBEDDING_MODEL_CONFIG,
    ) -> None:
        self._config = model_config
        self._model: "SentenceTransformer | None" = None  # noqa: F821
        logger.debug(
            "Initialized SentenceTransformerEmbeddingGenerator with model=%s "
            "(lazy load, not yet loaded).",
            self._config.model_name,
        )

    @property
    def config(self) -> EmbeddingModelConfig:
        """Return the :class:`EmbeddingModelConfig` used by this generator."""
        return self._config

    def _ensure_model_loaded(self) -> "SentenceTransformer":  # noqa: F821
        """
        Lazily load and cache the underlying ``SentenceTransformer`` model.

        Returns
        -------
        SentenceTransformer
            The loaded model instance.

        Raises
        ------
        EmbeddingGenerationError
            If the ``sentence-transformers`` package is not installed, or
            the model fails to load (e.g. invalid model name, no network
            access on first download, corrupted cache).
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed; cannot load "
                "model '%s'.",
                self._config.model_name,
            )
            raise EmbeddingGenerationError(
                "The 'sentence-transformers' package is required to use "
                "SentenceTransformerEmbeddingGenerator. Install it via "
                "'pip install sentence-transformers'."
            ) from exc

        try:
            logger.info(
                "Loading embedding model '%s' on device='%s'...",
                self._config.model_name,
                self._config.device,
            )
            device = None if self._config.device == "auto" else self._config.device
            model = SentenceTransformer(self._config.model_name, device=device)
        except Exception as exc:
            logger.exception(
                "Failed to load embedding model '%s'.", self._config.model_name
            )
            raise EmbeddingGenerationError(
                f"Failed to load embedding model '{self._config.model_name}': {exc}"
            ) from exc

        try:
            model.max_seq_length = self._config.max_sequence_length
        except Exception:
            # Not fatal: some models may not expose a mutable max_seq_length.
            logger.warning(
                "Could not set max_seq_length=%d on model '%s'; "
                "continuing with model default.",
                self._config.max_sequence_length,
                self._config.model_name,
            )

        self._model = model
        logger.info(
            "Embedding model '%s' loaded successfully.", self._config.model_name
        )
        return self._model

    def generate(self, text: str) -> list[float]:
        """
        Generate a single embedding vector for the given text.

        See :meth:`BaseEmbeddingGenerator.generate` for full contract.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        return self.generate_batch([text])[0]

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of texts.

        See :meth:`BaseEmbeddingGenerator.generate_batch` for full
        contract.
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"texts[{index}] must be a non-empty string."
                )

        model = self._ensure_model_loaded()

        try:
            logger.debug(
                "Encoding batch of %d text(s) with model '%s' "
                "(batch_size=%d, normalize=%s).",
                len(texts),
                self._config.model_name,
                self._config.batch_size,
                self._config.normalize_embeddings,
            )
            embeddings = model.encode(
                texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception(
                "Failed to generate embeddings for a batch of %d text(s).",
                len(texts),
            )
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings: {exc}"
            ) from exc

        return [vector.tolist() for vector in embeddings]


__all__: list[str] = [
    "DEFAULT_EMBEDDING_MODEL_CONFIG",
    "EmbeddingGenerationError",
    "BaseEmbeddingGenerator",
    "SentenceTransformerEmbeddingGenerator",
]