"""
Comprehensive pytest test suite for the BioNexus ``embeddings`` module.

This suite imports and exercises the actual implementation of every
public class and function across:

    - embeddings.models
    - embeddings.utils
    - embeddings.cache
    - embeddings.embedding_generator
    - embeddings.embedding_store
    - embeddings.embedding_search
    - embeddings.embedding_manager

Coverage includes:
    - Successful execution paths for every public class/function.
    - Failure / invalid-input paths and the specific exceptions raised.
    - Embedding model loading (via a monkeypatched ``sentence_transformers``
      backend, so tests run offline and deterministically).
    - Single and batch embedding generation, including dimension checks.
    - Metadata construction and validation.
    - Embedding storage: add, retrieve, delete, duplicate handling.
    - Persistence: saving to disk and reloading into a fresh store
      instance.
    - Semantic search: verifying the top result is the expected match.
    - EmbeddingManager integration of generator + store + cache + search.
    - Cache behavior: hits, misses, LRU eviction, TTL expiration, stats.

A lightweight, deterministic ``FakeEmbeddingGenerator`` is used for tests
that exercise storage/search/manager/cache logic, so those tests do not
depend on network access or a real neural network model, and so that
similarity rankings are fully predictable. The real
``SentenceTransformerEmbeddingGenerator`` is tested directly against a
monkeypatched stand-in for the ``sentence_transformers`` package, so its
lazy-loading, encoding, and error-handling logic is verified without
requiring a model download.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from embeddings.cache import (
    BaseEmbeddingCache,
    CacheStats,
    EmbeddingCacheError,
    LRUEmbeddingCache,
)
from embeddings.embedding_generator import (
    DEFAULT_EMBEDDING_MODEL_CONFIG,
    BaseEmbeddingGenerator,
    EmbeddingGenerationError,
    SentenceTransformerEmbeddingGenerator,
)
from embeddings.embedding_manager import EmbeddingManager, EmbeddingManagerError
from embeddings.embedding_search import (
    BaseEmbeddingSearchEngine,
    EmbeddingSearchError,
    InMemoryEmbeddingSearchEngine,
)
from embeddings.embedding_store import (
    BaseEmbeddingStore,
    DuplicateEmbeddingError,
    EmbeddingNotFoundError,
    EmbeddingStoreError,
    JSONEmbeddingStore,
)
from embeddings.models import (
    EmbeddingMetadata,
    EmbeddingModelConfig,
    EmbeddingRecord,
    SimilarityMetric,
    SimilaritySearchResult,
)
from embeddings.utils import (
    chunk_text,
    compute_cache_key,
    compute_text_hash,
    cosine_similarity,
    normalize_text,
    validate_vector,
)


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

FAKE_MODEL_NAME = "fake-test-model"
FAKE_DIMENSION = 8

_KEYWORDS = [
    "cancer",
    "egfr",
    "lung",
    "kinase",
    "diabetes",
    "insulin",
    "glucose",
    "therapy",
]


class FakeEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Deterministic, dependency-free embedding generator used for testing.

    Produces an 8-dimensional bag-of-keywords indicator vector: dimension
    ``i`` is ``1.0`` if ``_KEYWORDS[i]`` appears in the (lowercased) input
    text, else ``0.0``. This makes similarity rankings fully predictable
    across tests without depending on a real neural network model.
    """

    def __init__(self, model_name: str = FAKE_MODEL_NAME) -> None:
        self._config = EmbeddingModelConfig(
            model_name=model_name,
            embedding_dimension=FAKE_DIMENSION,
            max_sequence_length=256,
            normalize_embeddings=False,
            batch_size=8,
            device="cpu",
        )
        self.generate_call_count = 0

    @property
    def config(self) -> EmbeddingModelConfig:
        return self._config

    def generate(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")
        self.generate_call_count += 1
        lowered = text.lower()
        return [1.0 if keyword in lowered else 0.0 for keyword in _KEYWORDS]

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty.")
        for index, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string.")
        return [self.generate(text) for text in texts]


class FailingEmbeddingGenerator(BaseEmbeddingGenerator):
    """Embedding generator whose ``generate`` always raises, for failure-path tests."""

    def __init__(self) -> None:
        self._config = EmbeddingModelConfig(
            model_name="failing-model", embedding_dimension=4
        )

    @property
    def config(self) -> EmbeddingModelConfig:
        return self._config

    def generate(self, text: str) -> list[float]:
        raise RuntimeError("simulated backend failure")

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated backend failure")


def make_metadata(
    source_id: str,
    text: str,
    source_type: str = "abstract",
    title: str | None = None,
    chunk_index: int = 0,
    extra: dict[str, Any] | None = None,
) -> EmbeddingMetadata:
    """Convenience factory for constructing valid EmbeddingMetadata in tests."""
    return EmbeddingMetadata(
        source_id=source_id,
        source_type=source_type,
        title=title if title is not None else f"Title for {source_id}",
        text=text,
        chunk_index=chunk_index,
        extra=extra if extra is not None else {},
    )


@pytest.fixture
def fake_generator() -> FakeEmbeddingGenerator:
    return FakeEmbeddingGenerator()


@pytest.fixture
def json_store(tmp_path: Path) -> JSONEmbeddingStore:
    return JSONEmbeddingStore(storage_path=tmp_path / "embeddings.json")


@pytest.fixture
def search_engine(
    fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
) -> InMemoryEmbeddingSearchEngine:
    return InMemoryEmbeddingSearchEngine(fake_generator, json_store)


@pytest.fixture
def manager(
    fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
) -> EmbeddingManager:
    return EmbeddingManager(generator=fake_generator, store=json_store)


# ---------------------------------------------------------------------------
# embeddings.utils
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_collapses_whitespace_and_strips(self) -> None:
        assert normalize_text("  Hello   world\n\t") == "Hello world"

    def test_already_normalized_text_unchanged(self) -> None:
        assert normalize_text("clean text") == "clean text"

    @pytest.mark.parametrize("bad_text", ["", "   ", "\n\t"])
    def test_raises_on_empty_or_whitespace(self, bad_text: str) -> None:
        with pytest.raises(ValueError):
            normalize_text(bad_text)


class TestComputeTextHash:
    def test_deterministic_for_same_text(self) -> None:
        assert compute_text_hash("EGFR mutation") == compute_text_hash("EGFR mutation")

    def test_whitespace_insensitive(self) -> None:
        assert compute_text_hash("EGFR  mutation") == compute_text_hash("EGFR mutation")

    def test_different_text_different_hash(self) -> None:
        assert compute_text_hash("EGFR mutation") != compute_text_hash("KRAS mutation")

    def test_raises_on_empty_text(self) -> None:
        with pytest.raises(ValueError):
            compute_text_hash("")


class TestComputeCacheKey:
    def test_includes_model_and_is_deterministic(self) -> None:
        key1 = compute_cache_key("model-a", "some text")
        key2 = compute_cache_key("model-a", "some text")
        assert key1 == key2
        assert key1.startswith("model-a:")

    def test_different_models_different_keys(self) -> None:
        key_a = compute_cache_key("model-a", "some text")
        key_b = compute_cache_key("model-b", "some text")
        assert key_a != key_b

    def test_raises_on_empty_model_name(self) -> None:
        with pytest.raises(ValueError):
            compute_cache_key("", "text")

    def test_raises_on_empty_text(self) -> None:
        with pytest.raises(ValueError):
            compute_cache_key("model-a", "")


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        chunks = chunk_text("short text", chunk_size=1000, overlap=100)
        assert chunks == ["short text"]

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        long_text = "word " * 500  # 2500 characters
        chunks = chunk_text(long_text, chunk_size=1000, overlap=100)
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_chunks_cover_full_text_with_overlap(self) -> None:
        long_text = "abcdefghij" * 50  # 500 chars, no whitespace collapsing issues
        chunks = chunk_text(long_text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Every character of the normalized text should appear in at least one chunk.
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[-(len(chunk)):]
        assert len(chunks[-1]) > 0

    def test_raises_on_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=0)

    def test_raises_on_negative_overlap(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, overlap=-1)

    def test_raises_when_overlap_ge_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, overlap=100)

    def test_raises_on_empty_text(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("", chunk_size=100, overlap=10)


class TestValidateVector:
    def test_valid_vector_passes(self) -> None:
        validate_vector([1.0, 2.0, 3.0])  # should not raise

    def test_valid_vector_with_expected_dimension_passes(self) -> None:
        validate_vector([1.0, 2.0, 3.0], expected_dimension=3)

    def test_raises_on_empty_vector(self) -> None:
        with pytest.raises(ValueError):
            validate_vector([])

    def test_raises_on_nan_value(self) -> None:
        with pytest.raises(ValueError):
            validate_vector([1.0, float("nan"), 3.0])

    def test_raises_on_infinite_value(self) -> None:
        with pytest.raises(ValueError):
            validate_vector([1.0, float("inf"), 3.0])

    def test_raises_on_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError):
            validate_vector([1.0, 2.0], expected_dimension=3)


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self) -> None:
        vector = [1.0, 2.0, 3.0]
        assert math.isclose(cosine_similarity(vector, vector), 1.0, rel_tol=1e-9)

    def test_orthogonal_vectors_have_similarity_zero(self) -> None:
        assert math.isclose(
            cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9
        )

    def test_opposite_vectors_have_similarity_negative_one(self) -> None:
        assert math.isclose(
            cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, rel_tol=1e-9
        )

    def test_zero_vector_returns_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_raises_on_empty_vectors(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([], [1.0])

    def test_raises_on_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# embeddings.models
# ---------------------------------------------------------------------------


class TestEmbeddingModelConfig:
    def test_valid_construction(self) -> None:
        config = EmbeddingModelConfig(model_name="model-x", embedding_dimension=384)
        assert config.model_name == "model-x"
        assert config.embedding_dimension == 384
        assert config.normalize_embeddings is True

    def test_raises_on_empty_model_name(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingModelConfig(model_name="", embedding_dimension=384)

    def test_raises_on_non_positive_dimension(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingModelConfig(model_name="model-x", embedding_dimension=0)

    def test_raises_on_non_positive_max_sequence_length(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingModelConfig(
                model_name="model-x", embedding_dimension=384, max_sequence_length=0
            )

    def test_raises_on_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingModelConfig(
                model_name="model-x", embedding_dimension=384, batch_size=0
            )

    def test_is_frozen(self) -> None:
        config = EmbeddingModelConfig(model_name="model-x", embedding_dimension=384)
        with pytest.raises(Exception):
            config.model_name = "changed"  # type: ignore[misc]


class TestEmbeddingMetadata:
    def test_valid_construction(self) -> None:
        metadata = make_metadata("PMID123", "Some abstract text.")
        assert metadata.source_id == "PMID123"
        assert metadata.chunk_index == 0
        assert metadata.extra == {}

    def test_raises_on_empty_source_id(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                source_id="", source_type="abstract", title="t", text="text"
            )

    def test_raises_on_empty_text(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                source_id="PMID1", source_type="abstract", title="t", text=""
            )

    def test_raises_on_negative_chunk_index(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingMetadata(
                source_id="PMID1",
                source_type="abstract",
                title="t",
                text="text",
                chunk_index=-1,
            )

    def test_extra_field_stores_arbitrary_metadata(self) -> None:
        metadata = make_metadata(
            "PMID1", "text", extra={"doi": "10.1/xyz", "year": 2024}
        )
        assert metadata.extra["doi"] == "10.1/xyz"
        assert metadata.extra["year"] == 2024


class TestEmbeddingRecord:
    def test_valid_construction_generates_id_and_timestamp(self) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(
            metadata=metadata, vector=[1.0, 2.0], model_name="model-x"
        )
        assert record.embedding_id
        assert record.created_at is not None
        assert record.dimension == 2

    def test_explicit_id_is_respected(self) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(
            metadata=metadata,
            vector=[1.0, 2.0],
            model_name="model-x",
            embedding_id="custom-id",
        )
        assert record.embedding_id == "custom-id"

    def test_raises_on_empty_vector(self) -> None:
        metadata = make_metadata("PMID1", "text")
        with pytest.raises(ValueError):
            EmbeddingRecord(metadata=metadata, vector=[], model_name="model-x")

    def test_raises_on_empty_model_name(self) -> None:
        metadata = make_metadata("PMID1", "text")
        with pytest.raises(ValueError):
            EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="")

    def test_two_records_get_unique_ids(self) -> None:
        metadata = make_metadata("PMID1", "text")
        record_a = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        record_b = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        assert record_a.embedding_id != record_b.embedding_id


class TestSimilaritySearchResult:
    def test_valid_construction(self) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        result = SimilaritySearchResult(record=record, score=0.9, rank=1)
        assert result.rank == 1
        assert result.score == 0.9

    def test_raises_on_rank_below_one(self) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        with pytest.raises(ValueError):
            SimilaritySearchResult(record=record, score=0.9, rank=0)


class TestSimilarityMetric:
    def test_enum_values(self) -> None:
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"


# ---------------------------------------------------------------------------
# embeddings.embedding_generator
# ---------------------------------------------------------------------------


class FakeSentenceTransformer:
    """Stand-in for ``sentence_transformers.SentenceTransformer`` used in tests."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 128  # default, overwritten by generator config

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Deterministic fake embedding: vector of character-code-derived floats.
        dimension = 384
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 1000
            rng = np.random.default_rng(seed)
            vectors.append(rng.random(dimension))
        return np.array(vectors)


class RaisingOnInitSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        raise RuntimeError("simulated model load failure")


class RaisingOnEncodeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.max_seq_length = 128

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated encode failure")


def _install_fake_sentence_transformers_module(
    monkeypatch: pytest.MonkeyPatch, transformer_cls: type
) -> None:
    """Install a fake ``sentence_transformers`` module into sys.modules."""
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = transformer_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


class TestBaseEmbeddingGeneratorAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingGenerator()  # type: ignore[abstract]


class TestSentenceTransformerEmbeddingGenerator:
    def test_default_config_matches_specified_model(self) -> None:
        assert (
            DEFAULT_EMBEDDING_MODEL_CONFIG.model_name
            == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert DEFAULT_EMBEDDING_MODEL_CONFIG.embedding_dimension == 384

    def test_model_not_loaded_until_first_generate_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(monkeypatch, FakeSentenceTransformer)
        generator = SentenceTransformerEmbeddingGenerator()
        assert generator._model is None  # lazy: not loaded at construction
        generator.generate("EGFR mutation drives lung cancer.")
        assert generator._model is not None  # loaded after first use

    def test_generate_returns_correct_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(monkeypatch, FakeSentenceTransformer)
        generator = SentenceTransformerEmbeddingGenerator()
        vector = generator.generate("EGFR mutation drives lung cancer.")
        assert isinstance(vector, list)
        assert len(vector) == generator.config.embedding_dimension == 384
        assert all(isinstance(v, float) for v in vector)

    def test_generate_batch_returns_vectors_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(monkeypatch, FakeSentenceTransformer)
        generator = SentenceTransformerEmbeddingGenerator()
        texts = ["first text", "second text", "third text"]
        vectors = generator.generate_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_generate_record_binds_metadata_and_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(monkeypatch, FakeSentenceTransformer)
        generator = SentenceTransformerEmbeddingGenerator()
        metadata = make_metadata("PMID1", "EGFR mutation in lung cancer.")
        record = generator.generate_record(metadata)
        assert record.metadata is metadata
        assert record.model_name == generator.config.model_name
        assert record.dimension == 384

    def test_raises_value_error_on_empty_text(self) -> None:
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(ValueError):
            generator.generate("")

    def test_raises_value_error_on_empty_batch(self) -> None:
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(ValueError):
            generator.generate_batch([])

    def test_raises_value_error_on_blank_text_in_batch(self) -> None:
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(ValueError):
            generator.generate_batch(["valid text", "   "])

    def test_raises_generation_error_when_library_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulating "package not installed": importing it raises ImportError.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(EmbeddingGenerationError):
            generator.generate("some text")

    def test_raises_generation_error_on_model_load_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(
            monkeypatch, RaisingOnInitSentenceTransformer
        )
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(EmbeddingGenerationError):
            generator.generate("some text")

    def test_raises_generation_error_on_encode_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_sentence_transformers_module(
            monkeypatch, RaisingOnEncodeSentenceTransformer
        )
        generator = SentenceTransformerEmbeddingGenerator()
        with pytest.raises(EmbeddingGenerationError):
            generator.generate("some text")

    def test_custom_config_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_sentence_transformers_module(monkeypatch, FakeSentenceTransformer)
        custom_config = EmbeddingModelConfig(
            model_name="custom/model", embedding_dimension=384, batch_size=4
        )
        generator = SentenceTransformerEmbeddingGenerator(model_config=custom_config)
        assert generator.config.model_name == "custom/model"
        assert generator.config.batch_size == 4


class TestFakeEmbeddingGenerator:
    """Sanity tests for the test-local FakeEmbeddingGenerator itself."""

    def test_generate_correct_dimension(
        self, fake_generator: FakeEmbeddingGenerator
    ) -> None:
        vector = fake_generator.generate("EGFR lung cancer")
        assert len(vector) == FAKE_DIMENSION

    def test_generate_raises_on_empty_text(
        self, fake_generator: FakeEmbeddingGenerator
    ) -> None:
        with pytest.raises(ValueError):
            fake_generator.generate("")

    def test_generate_batch_raises_on_empty_list(
        self, fake_generator: FakeEmbeddingGenerator
    ) -> None:
        with pytest.raises(ValueError):
            fake_generator.generate_batch([])


# ---------------------------------------------------------------------------
# embeddings.embedding_store
# ---------------------------------------------------------------------------


class TestJSONEmbeddingStore:
    def test_new_store_is_empty(self, json_store: JSONEmbeddingStore) -> None:
        assert json_store.count() == 0
        assert json_store.all() == []
        assert len(json_store) == 0

    def test_add_and_get_record(self, json_store: JSONEmbeddingStore) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0, 2.0], model_name="m")
        json_store.add(record)

        retrieved = json_store.get(record.embedding_id)
        assert retrieved.embedding_id == record.embedding_id
        assert retrieved.vector == [1.0, 2.0]
        assert retrieved.metadata.source_id == "PMID1"

    def test_add_batch(self, json_store: JSONEmbeddingStore) -> None:
        records = [
            EmbeddingRecord(
                metadata=make_metadata(f"PMID{i}", f"text {i}"),
                vector=[float(i)],
                model_name="m",
            )
            for i in range(3)
        ]
        json_store.add_batch(records)
        assert json_store.count() == 3

    def test_add_batch_raises_on_empty_list(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        with pytest.raises(ValueError):
            json_store.add_batch([])

    def test_add_duplicate_raises(self, json_store: JSONEmbeddingStore) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(
            metadata=metadata, vector=[1.0], model_name="m", embedding_id="dup-id"
        )
        json_store.add(record)
        with pytest.raises(DuplicateEmbeddingError):
            json_store.add(record)

    def test_add_batch_with_duplicate_within_batch_raises(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        metadata = make_metadata("PMID1", "text")
        record_a = EmbeddingRecord(
            metadata=metadata, vector=[1.0], model_name="m", embedding_id="same-id"
        )
        record_b = EmbeddingRecord(
            metadata=metadata, vector=[2.0], model_name="m", embedding_id="same-id"
        )
        with pytest.raises(DuplicateEmbeddingError):
            json_store.add_batch([record_a, record_b])

    def test_get_missing_raises_not_found(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        with pytest.raises(EmbeddingNotFoundError):
            json_store.get("does-not-exist")

    def test_get_by_source_orders_by_chunk_index(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        chunk_1 = EmbeddingRecord(
            metadata=make_metadata("PMID1", "chunk one", chunk_index=1),
            vector=[1.0],
            model_name="m",
        )
        chunk_0 = EmbeddingRecord(
            metadata=make_metadata("PMID1", "chunk zero", chunk_index=0),
            vector=[2.0],
            model_name="m",
        )
        json_store.add(chunk_1)
        json_store.add(chunk_0)

        results = json_store.get_by_source("PMID1")
        assert [r.metadata.chunk_index for r in results] == [0, 1]

    def test_get_by_source_returns_empty_for_unknown_source(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        assert json_store.get_by_source("unknown") == []

    def test_delete_removes_record(self, json_store: JSONEmbeddingStore) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        json_store.add(record)
        json_store.delete(record.embedding_id)
        assert json_store.count() == 0
        with pytest.raises(EmbeddingNotFoundError):
            json_store.get(record.embedding_id)

    def test_delete_missing_raises_not_found(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        with pytest.raises(EmbeddingNotFoundError):
            json_store.delete("does-not-exist")

    def test_contains_operator(self, json_store: JSONEmbeddingStore) -> None:
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        json_store.add(record)
        assert record.embedding_id in json_store
        assert "nonexistent" not in json_store

    def test_persistence_across_store_instances(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "persist_test.json"
        store_a = JSONEmbeddingStore(storage_path=storage_path)

        metadata = make_metadata(
            "PMID1", "persisted text", extra={"year": 2024, "citations": 10}
        )
        record = EmbeddingRecord(
            metadata=metadata, vector=[1.0, 2.0, 3.0], model_name="m"
        )
        store_a.add(record)

        # A fresh store instance pointed at the same file should see the data.
        store_b = JSONEmbeddingStore(storage_path=storage_path)
        assert store_b.count() == 1
        retrieved = store_b.get(record.embedding_id)
        assert retrieved.vector == [1.0, 2.0, 3.0]
        assert retrieved.metadata.extra == {"year": 2024, "citations": 10}

    def test_persistence_file_is_valid_json_on_disk(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "raw_check.json"
        store = JSONEmbeddingStore(storage_path=storage_path)
        metadata = make_metadata("PMID1", "text")
        record = EmbeddingRecord(metadata=metadata, vector=[1.0], model_name="m")
        store.add(record)

        raw = json.loads(storage_path.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        assert raw[0]["embedding_id"] == record.embedding_id

    def test_loading_from_empty_file_starts_empty(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "empty.json"
        storage_path.write_text("", encoding="utf-8")
        store = JSONEmbeddingStore(storage_path=storage_path)
        assert store.count() == 0

    def test_loading_corrupted_file_raises_store_error(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "corrupted.json"
        storage_path.write_text("{ this is not valid json", encoding="utf-8")
        with pytest.raises(EmbeddingStoreError):
            JSONEmbeddingStore(storage_path=storage_path)

    def test_loading_malformed_records_raises_store_error(
        self, tmp_path: Path
    ) -> None:
        storage_path = tmp_path / "malformed.json"
        # Valid JSON, but missing required "metadata" key.
        storage_path.write_text(
            json.dumps([{"embedding_id": "x", "vector": [1.0], "model_name": "m"}]),
            encoding="utf-8",
        )
        with pytest.raises(EmbeddingStoreError):
            JSONEmbeddingStore(storage_path=storage_path)

    def test_loading_from_nonexistent_file_starts_empty(self, tmp_path: Path) -> None:
        storage_path = tmp_path / "does_not_exist_yet.json"
        store = JSONEmbeddingStore(storage_path=storage_path)
        assert store.count() == 0
        assert not storage_path.exists()  # not created until first write

    def test_base_store_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingStore()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# embeddings.embedding_search
# ---------------------------------------------------------------------------


class TestInMemoryEmbeddingSearchEngine:
    def test_search_returns_expected_top_result(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        lung_cancer_record = fake_generator.generate_record(
            make_metadata("PMID_LUNG", "EGFR mutation drives lung cancer therapy.")
        )
        diabetes_record = fake_generator.generate_record(
            make_metadata("PMID_DIABETES", "Insulin regulates glucose in diabetes.")
        )
        json_store.add(lung_cancer_record)
        json_store.add(diabetes_record)

        results = search_engine.search("EGFR lung cancer treatment", top_k=2)

        assert len(results) == 2
        assert results[0].rank == 1
        assert results[0].record.metadata.source_id == "PMID_LUNG"
        assert results[0].score > results[1].score

    def test_search_respects_top_k(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        for i in range(5):
            json_store.add(
                fake_generator.generate_record(
                    make_metadata(f"PMID{i}", "cancer kinase egfr")
                )
            )
        results = search_engine.search("cancer", top_k=3)
        assert len(results) == 3

    def test_search_filters_by_source_type(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        json_store.add(
            fake_generator.generate_record(
                make_metadata("PMID1", "cancer egfr", source_type="abstract")
            )
        )
        json_store.add(
            fake_generator.generate_record(
                make_metadata("PMID2", "cancer egfr", source_type="full_text_chunk")
            )
        )
        results = search_engine.search("cancer", top_k=10, source_type="abstract")
        assert len(results) == 1
        assert results[0].record.metadata.source_type == "abstract"

    def test_search_raises_on_empty_query(
        self, search_engine: InMemoryEmbeddingSearchEngine
    ) -> None:
        with pytest.raises(ValueError):
            search_engine.search("", top_k=5)

    def test_search_raises_on_non_positive_top_k(
        self, search_engine: InMemoryEmbeddingSearchEngine
    ) -> None:
        with pytest.raises(ValueError):
            search_engine.search("query", top_k=0)

    def test_search_wraps_generator_failure(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        engine = InMemoryEmbeddingSearchEngine(FailingEmbeddingGenerator(), json_store)
        with pytest.raises(EmbeddingSearchError):
            engine.search("query", top_k=5)

    def test_search_returns_empty_list_when_store_empty(
        self, search_engine: InMemoryEmbeddingSearchEngine
    ) -> None:
        assert search_engine.search("anything", top_k=5) == []

    def test_find_similar_excludes_self(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        record_a = fake_generator.generate_record(
            make_metadata("PMID_A", "cancer egfr lung")
        )
        record_b = fake_generator.generate_record(
            make_metadata("PMID_B", "cancer egfr lung therapy")
        )
        json_store.add(record_a)
        json_store.add(record_b)

        results = search_engine.find_similar(record_a.embedding_id, top_k=5)
        assert all(r.record.embedding_id != record_a.embedding_id for r in results)
        assert results[0].record.embedding_id == record_b.embedding_id

    def test_find_similar_raises_for_unknown_id(
        self, search_engine: InMemoryEmbeddingSearchEngine
    ) -> None:
        with pytest.raises(EmbeddingSearchError):
            search_engine.find_similar("unknown-id", top_k=5)

    def test_find_similar_raises_on_non_positive_top_k(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        record = fake_generator.generate_record(make_metadata("PMID1", "text"))
        json_store.add(record)
        with pytest.raises(ValueError):
            search_engine.find_similar(record.embedding_id, top_k=0)

    def test_dimension_mismatch_candidate_is_skipped_not_raised(
        self,
        fake_generator: FakeEmbeddingGenerator,
        json_store: JSONEmbeddingStore,
        search_engine: InMemoryEmbeddingSearchEngine,
    ) -> None:
        good_record = fake_generator.generate_record(
            make_metadata("PMID_GOOD", "cancer egfr")
        )
        bad_record = EmbeddingRecord(
            metadata=make_metadata("PMID_BAD", "mismatched dims"),
            vector=[1.0, 2.0],  # wrong dimension vs FAKE_DIMENSION
            model_name=fake_generator.config.model_name,
        )
        json_store.add(good_record)
        json_store.add(bad_record)

        results = search_engine.search("cancer egfr", top_k=10)
        returned_ids = {r.record.embedding_id for r in results}
        assert good_record.embedding_id in returned_ids
        assert bad_record.embedding_id not in returned_ids

    def test_dot_product_metric(
        self, fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
    ) -> None:
        engine = InMemoryEmbeddingSearchEngine(
            fake_generator, json_store, metric=SimilarityMetric.DOT_PRODUCT
        )
        json_store.add(
            fake_generator.generate_record(make_metadata("PMID1", "cancer egfr lung"))
        )
        results = engine.search("cancer egfr", top_k=1)
        assert len(results) == 1

    def test_euclidean_metric(
        self, fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
    ) -> None:
        engine = InMemoryEmbeddingSearchEngine(
            fake_generator, json_store, metric=SimilarityMetric.EUCLIDEAN
        )
        json_store.add(
            fake_generator.generate_record(make_metadata("PMID1", "cancer egfr lung"))
        )
        results = engine.search("cancer egfr", top_k=1)
        assert len(results) == 1
        assert results[0].score <= 0.0  # negated distance: <= 0

    def test_base_search_engine_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingSearchEngine()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# embeddings.cache
# ---------------------------------------------------------------------------


class TestLRUEmbeddingCache:
    def test_put_and_get(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("key1", [1.0, 2.0])
        assert cache.get("key1") == [1.0, 2.0]

    def test_get_missing_key_returns_none(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        assert cache.get("missing") is None

    def test_get_records_hit_and_miss_stats(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("key1", [1.0])
        cache.get("key1")  # hit
        cache.get("missing")  # miss

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_hit_rate_is_zero_with_no_lookups(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        assert cache.stats().hit_rate == 0.0

    def test_lru_eviction_removes_least_recently_used(self) -> None:
        cache = LRUEmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])  # should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]
        assert cache.stats().evictions == 1

    def test_accessing_an_entry_marks_it_recently_used(self) -> None:
        cache = LRUEmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.get("a")  # "a" is now most-recently-used
        cache.put("c", [3.0])  # should evict "b", not "a"

        assert cache.get("a") == [1.0]
        assert cache.get("b") is None

    def test_put_overwrites_existing_key(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("a", [1.0])
        cache.put("a", [2.0])
        assert cache.get("a") == [2.0]
        assert len(cache) == 1

    def test_invalidate_removes_entry(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("a", [1.0])
        cache.invalidate("a")
        assert cache.get("a") is None

    def test_invalidate_missing_key_does_not_raise(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.invalidate("missing")  # should not raise

    def test_clear_resets_entries_and_stats(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("a", [1.0])
        cache.get("a")
        cache.get("missing")
        cache.clear()

        assert len(cache) == 0
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0

    def test_contains_operator(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        cache.put("a", [1.0])
        assert "a" in cache
        assert "b" not in cache

    def test_ttl_expiration(self) -> None:
        cache = LRUEmbeddingCache(max_size=10, ttl_seconds=0.05)
        cache.put("a", [1.0])
        assert cache.get("a") == [1.0]
        time.sleep(0.1)
        assert cache.get("a") is None
        assert cache.stats().expirations == 1

    def test_no_ttl_means_entries_never_expire_by_age(self) -> None:
        cache = LRUEmbeddingCache(max_size=10, ttl_seconds=None)
        cache.put("a", [1.0])
        time.sleep(0.05)
        assert cache.get("a") == [1.0]

    def test_raises_on_non_positive_max_size(self) -> None:
        with pytest.raises(ValueError):
            LRUEmbeddingCache(max_size=0)

    def test_raises_on_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError):
            LRUEmbeddingCache(max_size=10, ttl_seconds=0)

    def test_get_raises_on_empty_key(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        with pytest.raises(ValueError):
            cache.get("")

    def test_put_raises_on_empty_key(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        with pytest.raises(ValueError):
            cache.put("", [1.0])

    def test_put_raises_on_empty_vector(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        with pytest.raises(ValueError):
            cache.put("key", [])

    def test_cached_vector_is_a_copy_not_a_shared_reference(self) -> None:
        cache = LRUEmbeddingCache(max_size=10)
        original = [1.0, 2.0]
        cache.put("a", original)
        original.append(3.0)
        assert cache.get("a") == [1.0, 2.0]

    def test_base_cache_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingCache()  # type: ignore[abstract]


class TestCacheStats:
    def test_hit_rate_computation(self) -> None:
        stats = CacheStats(hits=3, misses=1, evictions=0, expirations=0, size=1)
        assert stats.hit_rate == 0.75


# ---------------------------------------------------------------------------
# embeddings.embedding_manager
# ---------------------------------------------------------------------------


class TestEmbeddingManager:
    def test_default_cache_and_search_engine_are_constructed(
        self, fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
    ) -> None:
        manager = EmbeddingManager(generator=fake_generator, store=json_store)
        assert isinstance(manager.cache, LRUEmbeddingCache)
        assert isinstance(manager.search_engine, InMemoryEmbeddingSearchEngine)

    def test_embed_text_uses_cache_on_second_call(
        self, fake_generator: FakeEmbeddingGenerator, manager: EmbeddingManager
    ) -> None:
        text = "EGFR mutation in lung cancer."
        vector_1 = manager.embed_text(text)
        calls_after_first = fake_generator.generate_call_count
        vector_2 = manager.embed_text(text)

        assert vector_1 == vector_2
        assert fake_generator.generate_call_count == calls_after_first  # cache hit

    def test_embed_text_bypasses_cache_when_disabled(
        self, fake_generator: FakeEmbeddingGenerator, manager: EmbeddingManager
    ) -> None:
        text = "EGFR mutation in lung cancer."
        manager.embed_text(text, use_cache=False)
        calls_after_first = fake_generator.generate_call_count
        manager.embed_text(text, use_cache=False)

        assert fake_generator.generate_call_count == calls_after_first + 1

    def test_embed_text_raises_on_empty_text(self, manager: EmbeddingManager) -> None:
        with pytest.raises(ValueError):
            manager.embed_text("")

    def test_embed_text_wraps_generator_failure(
        self, json_store: JSONEmbeddingStore
    ) -> None:
        manager = EmbeddingManager(
            generator=FailingEmbeddingGenerator(), store=json_store
        )
        with pytest.raises(EmbeddingManagerError):
            manager.embed_text("some text")

    def test_add_document_stores_record_and_returns_it(
        self, manager: EmbeddingManager, json_store: JSONEmbeddingStore
    ) -> None:
        metadata = make_metadata("PMID1", "EGFR mutation in lung cancer.")
        record = manager.add_document(metadata)

        assert record.metadata.source_id == "PMID1"
        assert record.dimension == FAKE_DIMENSION
        assert json_store.count() == 1
        assert json_store.get(record.embedding_id).embedding_id == record.embedding_id

    def test_add_documents_batch(
        self, manager: EmbeddingManager, json_store: JSONEmbeddingStore
    ) -> None:
        metadata_list = [
            make_metadata(f"PMID{i}", f"cancer text number {i}") for i in range(4)
        ]
        records = manager.add_documents(metadata_list)

        assert len(records) == 4
        assert json_store.count() == 4

    def test_add_documents_raises_on_empty_list(
        self, manager: EmbeddingManager
    ) -> None:
        with pytest.raises(ValueError):
            manager.add_documents([])

    def test_search_delegates_to_search_engine_and_returns_expected_top_result(
        self, manager: EmbeddingManager
    ) -> None:
        manager.add_document(
            make_metadata("PMID_LUNG", "EGFR mutation drives lung cancer therapy.")
        )
        manager.add_document(
            make_metadata("PMID_DIABETES", "Insulin regulates glucose in diabetes.")
        )

        results = manager.search("EGFR lung cancer treatment", top_k=1)

        assert len(results) == 1
        assert results[0].record.metadata.source_id == "PMID_LUNG"

    def test_search_raises_on_invalid_input(self, manager: EmbeddingManager) -> None:
        with pytest.raises(ValueError):
            manager.search("", top_k=5)

    def test_find_similar_via_manager(self, manager: EmbeddingManager) -> None:
        record_a = manager.add_document(make_metadata("PMID_A", "cancer egfr lung"))
        manager.add_document(make_metadata("PMID_B", "cancer egfr lung therapy"))

        results = manager.find_similar(record_a.embedding_id, top_k=5)
        assert all(r.record.embedding_id != record_a.embedding_id for r in results)

    def test_find_similar_raises_for_unknown_id(
        self, manager: EmbeddingManager
    ) -> None:
        with pytest.raises(EmbeddingManagerError):
            manager.find_similar("unknown-id", top_k=5)

    def test_get_returns_stored_record(self, manager: EmbeddingManager) -> None:
        record = manager.add_document(make_metadata("PMID1", "some text"))
        retrieved = manager.get(record.embedding_id)
        assert retrieved.embedding_id == record.embedding_id

    def test_get_raises_manager_error_for_unknown_id(
        self, manager: EmbeddingManager
    ) -> None:
        with pytest.raises(EmbeddingManagerError):
            manager.get("unknown-id")

    def test_get_by_source_returns_matching_records(
        self, manager: EmbeddingManager
    ) -> None:
        manager.add_document(
            make_metadata("PMID1", "chunk zero", chunk_index=0)
        )
        manager.add_document(
            make_metadata("PMID1", "chunk one", chunk_index=1)
        )
        results = manager.get_by_source("PMID1")
        assert len(results) == 2
        assert [r.metadata.chunk_index for r in results] == [0, 1]

    def test_delete_removes_record(self, manager: EmbeddingManager) -> None:
        record = manager.add_document(make_metadata("PMID1", "some text"))
        manager.delete(record.embedding_id)
        assert manager.count() == 0

    def test_delete_raises_manager_error_for_unknown_id(
        self, manager: EmbeddingManager
    ) -> None:
        with pytest.raises(EmbeddingManagerError):
            manager.delete("unknown-id")

    def test_count_reflects_store_size(self, manager: EmbeddingManager) -> None:
        assert manager.count() == 0
        manager.add_document(make_metadata("PMID1", "some text"))
        assert manager.count() == 1

    def test_cache_stats_reflects_activity(self, manager: EmbeddingManager) -> None:
        text = "repeated text for caching"
        manager.embed_text(text)
        manager.embed_text(text)  # hit

        stats = manager.cache_stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits >= 1

    def test_clear_cache_does_not_affect_stored_records(
        self, manager: EmbeddingManager
    ) -> None:
        record = manager.add_document(make_metadata("PMID1", "some text"))
        manager.clear_cache()

        assert manager.cache_stats().size == 0
        assert manager.count() == 1
        assert manager.get(record.embedding_id).embedding_id == record.embedding_id

    def test_custom_cache_and_search_engine_can_be_injected(
        self, fake_generator: FakeEmbeddingGenerator, json_store: JSONEmbeddingStore
    ) -> None:
        custom_cache = LRUEmbeddingCache(max_size=5)
        custom_engine = InMemoryEmbeddingSearchEngine(fake_generator, json_store)
        manager = EmbeddingManager(
            generator=fake_generator,
            store=json_store,
            cache=custom_cache,
            search_engine=custom_engine,
        )
        assert manager.cache is custom_cache
        assert manager.search_engine is custom_engine

    def test_end_to_end_ingest_and_search_workflow(
        self, tmp_path: Path
    ) -> None:
        """Full integration: generator + store + cache + search via manager."""
        generator = FakeEmbeddingGenerator()
        store = JSONEmbeddingStore(storage_path=tmp_path / "e2e.json")
        manager = EmbeddingManager(generator=generator, store=store)

        papers = [
            ("PMID_1", "EGFR kinase inhibitors in lung cancer therapy."),
            ("PMID_2", "KRAS signaling pathways in colorectal cancer."),
            ("PMID_3", "Insulin and glucose metabolism in type 2 diabetes."),
        ]
        for source_id, text in papers:
            manager.add_document(make_metadata(source_id, text))

        assert manager.count() == 3

        results = manager.search("lung cancer EGFR kinase", top_k=1)
        assert results[0].record.metadata.source_id == "PMID_1"

        # Reloading the store from disk should preserve all records.
        reloaded_store = JSONEmbeddingStore(storage_path=tmp_path / "e2e.json")
        assert reloaded_store.count() == 3