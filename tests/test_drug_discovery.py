"""
tests/test_drug_discovery.py

Comprehensive test suite for the BioNexus Drug Discovery module.

This module contains unit and integration tests covering all major
components of the Drug Discovery pipeline, including molecular property
calculation, compound similarity search, target prediction, toxicity
prediction, drug repurposing, candidate ranking, caching, utilities, and
the complete pipeline orchestration.

Test coverage targets:
    - All public APIs are tested
    - Valid and invalid inputs are handled correctly
    - Edge cases and empty inputs are tested
    - Cache behavior is verified
    - Ranking correctness is validated
    - Pipeline execution is tested end-to-end
    - Exception handling is thoroughly tested

Compatibility
-------------
Targets Python 3.11 with pytest.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import modules under test - use aliases to avoid naming conflicts
from drug_discovery import cache as cache_module
from drug_discovery import candidate_ranking as candidate_ranking_module
from drug_discovery import compound_similarity as compound_similarity_module
from drug_discovery import drug_repurposing as drug_repurposing_module
from drug_discovery import models as models_module
from drug_discovery import molecular_properties as molecular_properties_module
from drug_discovery import pipeline as pipeline_module
from drug_discovery import target_prediction as target_prediction_module
from drug_discovery import toxicity_prediction as toxicity_prediction_module
from drug_discovery import utils as utils_module

# Import specific classes/functions for convenience
from drug_discovery.models import (
    CandidateRanking,
    CandidateStatus,
    Compound,
    DiscoveryStage,
    DrugCandidate,
    DrugDiscoveryConfig,
    DrugDiscoveryResult,
    DrugTarget,
    MolecularProperties,
    PredictionConfidence,
    SimilarCompound,
    SimilarityMetric,
    TargetPrediction,
    TargetType,
    ToxicityLevel,
    ToxicityPrediction,
)
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
    MolecularPropertyCalculator,
    calculate_molecular_properties,
    validate_smiles,
)
from drug_discovery.compound_similarity import (
    CompoundSimilaritySearchEngine,
    FingerprintConfig,
    find_similar_compounds,
    generate_morgan_fingerprint,
)
from drug_discovery.target_prediction import (
    RuleBasedTargetPredictor,
    TargetPredictionError,
    TargetPredictionService,
    predict_targets,
    predict_targets_batch,
)
from drug_discovery.toxicity_prediction import (
    RuleBasedToxicityPredictor,
    ToxicityPredictionError,
    ToxicityPredictionService,
    predict_toxicity,
    predict_toxicity_batch,
)
from drug_discovery.drug_repurposing import (
    DrugRepurposingService,
    RepurposingQuery,
    RepurposingScoringError,
    RuleBasedRepurposingScorer,
    find_repurposing_candidates,
)
from drug_discovery.candidate_ranking import (
    CandidateEvidenceBundle,
    CandidateRankingConfig,
    CandidateRankingEngine,
    CandidateRankingError,
    RankingFilter,
    RankingStrategy,
    WeightedSumRankingStrategy,
    rank_candidates,
)
from drug_discovery.cache import (
    CacheConfig,
    CacheEntry,
    LRUTTLCache,
    get_default_cache,
    make_cache_key,
    cached,
)
from drug_discovery.pipeline import (
    DrugDiscoveryPipeline,
    PipelineConfig,
    PipelineError,
    run_drug_discovery,
)
from drug_discovery.utils import (
    InputValidationError,
    is_plausible_smiles,
    validate_non_empty_string,
    validate_in_range,
    validate_probability,
    validate_positive_integer,
    format_molecular_weight,
    format_similarity_score,
    format_percentage,
    clamp,
    normalize_min_max,
    score_to_confidence,
    safe_divide,
    safe_log,
    safe_mean,
    compute_basic_statistics,
    generate_unique_id,
    utc_now,
    deep_merge_dicts,
    coerce_bool,
    chunked,
    deduplicate_preserving_order,
    truncate_text,
    timed,
    retry,
    RetryExhaustedError,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def aspirin_compound() -> Compound:
    """Return an aspirin compound fixture."""
    return Compound(
        compound_id="C_ASPIRIN",
        name="Aspirin",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        molecular_formula="C9H8O4",
        molecular_weight=180.16,
        source="test",
    )


@pytest.fixture
def ibuprofen_compound() -> Compound:
    """Return an ibuprofen compound fixture."""
    return Compound(
        compound_id="C_IBUPROFEN",
        name="Ibuprofen",
        smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        molecular_formula="C13H18O2",
        molecular_weight=206.28,
        source="test",
    )


@pytest.fixture
def paracetamol_compound() -> Compound:
    """Return a paracetamol compound fixture."""
    return Compound(
        compound_id="C_PARACETAMOL",
        name="Paracetamol",
        smiles="CC(=O)NC1=CC=C(C=C1)O",
        molecular_formula="C8H9NO2",
        molecular_weight=151.16,
        source="test",
    )


@pytest.fixture
def invalid_compound() -> Compound:
    """Return an invalid compound fixture."""
    return Compound(
        compound_id="C_INVALID",
        name="Invalid",
        smiles="invalid_smiles_string",
        source="test",
    )


@pytest.fixture
def sample_compounds(
    aspirin_compound: Compound,
    ibuprofen_compound: Compound,
    paracetamol_compound: Compound,
) -> list[Compound]:
    """Return a list of sample compounds."""
    return [aspirin_compound, ibuprofen_compound, paracetamol_compound]


@pytest.fixture
def drug_target() -> DrugTarget:
    """Return a drug target fixture."""
    return DrugTarget(
        target_id="TGT_COX",
        name="Cyclooxygenase",
        target_type=TargetType.ENZYME,
        organism="Homo sapiens",
        gene_symbol="PTGS2",
        description="Cyclooxygenase-2 enzyme",
    )


@pytest.fixture
def molecular_properties(aspirin_compound: Compound) -> MolecularProperties:
    """Return a molecular properties fixture."""
    return MolecularProperties(
        compound_id=aspirin_compound.compound_id,
        molecular_weight=180.16,
        logp=1.19,
        tpsa=63.6,
        h_bond_donors=1,
        h_bond_acceptors=4,
        rotatable_bonds=3,
        aromatic_rings=1,
        heavy_atom_count=13,
        lipinski_violations=0,
        qed_score=0.85,
    )


@pytest.fixture
def target_prediction(aspirin_compound: Compound, drug_target: DrugTarget) -> TargetPrediction:
    """Return a target prediction fixture."""
    return TargetPrediction(
        compound_id=aspirin_compound.compound_id,
        target=drug_target,
        confidence_score=0.85,
        confidence_level=PredictionConfidence.HIGH,
        prediction_method="test_engine",
    )


@pytest.fixture
def toxicity_prediction(aspirin_compound: Compound) -> ToxicityPrediction:
    """Return a toxicity prediction fixture."""
    return ToxicityPrediction(
        compound_id=aspirin_compound.compound_id,
        toxicity_level=ToxicityLevel.LOW,
        toxicity_score=0.15,
        endpoint_scores={"hepatotoxicity": 0.1, "cardiotoxicity": 0.2},
        confidence=PredictionConfidence.HIGH,
        prediction_method="test_engine",
    )


@pytest.fixture
def cache_config() -> CacheConfig:
    """Return a cache configuration fixture."""
    return CacheConfig(
        max_size=100,
        default_ttl_seconds=60,
        track_namespace_stats=True,
    )


@pytest.fixture
def lru_cache(cache_config: CacheConfig) -> LRUTTLCache:
    """Return an LRU TTL cache fixture."""
    return LRUTTLCache(config=cache_config)


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Return a pipeline configuration fixture."""
    config = PipelineConfig(
        top_k=10,
        enable_cache=False,  # Disable cache for deterministic tests
        enable_similarity=True,
        enable_target_prediction=True,
        enable_toxicity_prediction=True,
        enable_repurposing=True,
        skip_invalid_compounds=True,
        max_compound_errors=10,
        collect_timing=True,
    )
    config.discovery_config.similarity_threshold = 0.5
    config.discovery_config.min_target_confidence = 0.3
    return config


# =============================================================================
# Tests for models.py
# =============================================================================

class TestModels:
    """Test suite for data models."""

    def test_compound_validation_valid(self, aspirin_compound: Compound) -> None:
        """Test that a valid compound passes validation."""
        assert aspirin_compound.compound_id == "C_ASPIRIN"
        assert aspirin_compound.name == "Aspirin"
        assert aspirin_compound.smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_compound_validation_empty_id(self) -> None:
        """Test that compound with empty ID raises ValueError."""
        with pytest.raises(ValueError, match="compound_id must be a non-empty string"):
            Compound(compound_id="", name="Test", smiles="C")

    def test_compound_validation_empty_smiles(self) -> None:
        """Test that compound with empty SMILES raises ValueError."""
        with pytest.raises(ValueError, match="smiles must be a non-empty string"):
            Compound(compound_id="C1", name="Test", smiles="")

    def test_compound_validation_negative_molecular_weight(self) -> None:
        """Test that compound with negative molecular weight raises ValueError."""
        with pytest.raises(ValueError, match="molecular_weight cannot be negative"):
            Compound(compound_id="C1", name="Test", smiles="C", molecular_weight=-1.0)

    def test_similar_compound_validation_valid(self, aspirin_compound: Compound) -> None:
        """Test that a valid SimilarCompound passes validation."""
        similar = SimilarCompound(
            compound=aspirin_compound,
            similarity_score=0.85,
            metric=SimilarityMetric.TANIMOTO,
            rank=1,
        )
        assert similar.similarity_score == 0.85
        assert similar.rank == 1

    def test_similar_compound_validation_invalid_score(self, aspirin_compound: Compound) -> None:
        """Test that SimilarCompound with invalid score raises ValueError."""
        with pytest.raises(ValueError, match="similarity_score must be within"):
            SimilarCompound(
                compound=aspirin_compound,
                similarity_score=1.5,
                metric=SimilarityMetric.TANIMOTO,
            )

    def test_molecular_properties_validation_negative(self) -> None:
        """Test that MolecularProperties with negative values raises ValueError."""
        with pytest.raises(ValueError, match="molecular_weight cannot be negative"):
            MolecularProperties(
                compound_id="C1",
                molecular_weight=-1.0,
                logp=1.0,
                tpsa=50.0,
                h_bond_donors=1,
                h_bond_acceptors=2,
                rotatable_bonds=3,
                aromatic_rings=1,
                heavy_atom_count=10,
                lipinski_violations=0,
            )

    def test_molecular_properties_lipinski_compliant(self, molecular_properties: MolecularProperties) -> None:
        """Test that lipinski_compliant returns correct value."""
        assert molecular_properties.is_lipinski_compliant() is True

        # Create non-compliant properties
        non_compliant = MolecularProperties(
            compound_id="C1",
            molecular_weight=600.0,
            logp=6.0,
            tpsa=50.0,
            h_bond_donors=6,
            h_bond_acceptors=11,
            rotatable_bonds=3,
            aromatic_rings=1,
            heavy_atom_count=10,
            lipinski_violations=4,
        )
        assert non_compliant.is_lipinski_compliant() is False

    def test_drug_target_validation(self) -> None:
        """Test that DrugTarget validation works correctly."""
        target = DrugTarget(
            target_id="T1",
            name="Test Target",
            target_type=TargetType.PROTEIN,
        )
        assert target.target_id == "T1"

        with pytest.raises(ValueError, match="target_id must be a non-empty string"):
            DrugTarget(target_id="", name="Test", target_type=TargetType.PROTEIN)

    def test_target_prediction_validation(self, drug_target: DrugTarget) -> None:
        """Test that TargetPrediction validation works correctly."""
        with pytest.raises(ValueError, match="confidence_score must be within"):
            TargetPrediction(
                compound_id="C1",
                target=drug_target,
                confidence_score=1.5,
                confidence_level=PredictionConfidence.HIGH,
            )

    def test_toxicity_prediction_validation(self) -> None:
        """Test that ToxicityPrediction validation works correctly."""
        with pytest.raises(ValueError, match="toxicity_score must be within"):
            ToxicityPrediction(
                compound_id="C1",
                toxicity_level=ToxicityLevel.LOW,
                toxicity_score=1.5,
            )

    def test_drug_candidate_validation(self, aspirin_compound: Compound) -> None:
        """Test that DrugCandidate validation works correctly."""
        candidate = DrugCandidate(
            candidate_id="DC1",
            compound=aspirin_compound,
            overall_score=0.75,
        )
        assert candidate.candidate_id == "DC1"

        with pytest.raises(ValueError, match="overall_score must be within"):
            DrugCandidate(
                candidate_id="DC1",
                compound=aspirin_compound,
                overall_score=1.5,
            )

    def test_drug_candidate_top_target_prediction(self, aspirin_compound: Compound, drug_target: DrugTarget) -> None:
        """Test that top_target_prediction returns the highest confidence prediction."""
        predictions = [
            TargetPrediction(
                compound_id="C1",
                target=drug_target,
                confidence_score=0.7,
                confidence_level=PredictionConfidence.HIGH,
            ),
            TargetPrediction(
                compound_id="C1",
                target=DrugTarget(
                    target_id="T2",
                    name="Target 2",
                    target_type=TargetType.PROTEIN,
                ),
                confidence_score=0.9,
                confidence_level=PredictionConfidence.VERY_HIGH,
            ),
        ]
        candidate = DrugCandidate(
            candidate_id="DC1",
            compound=aspirin_compound,
            target_predictions=predictions,
        )
        top = candidate.top_target_prediction()
        assert top is not None
        assert top.confidence_score == 0.9

    def test_drug_discovery_config_validation(self) -> None:
        """Test that DrugDiscoveryConfig validation works correctly."""
        config = DrugDiscoveryConfig()
        assert config.similarity_threshold == 0.7

        with pytest.raises(ValueError, match="similarity_threshold must be within"):
            DrugDiscoveryConfig(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="max_similar_compounds must be a positive integer"):
            DrugDiscoveryConfig(max_similar_compounds=0)

    def test_drug_discovery_result_ranked_candidates(self, aspirin_compound: Compound) -> None:
        """Test that ranked_candidates returns candidates in correct order."""
        result = DrugDiscoveryResult(
            run_id="R1",
            candidates=[
                DrugCandidate(candidate_id="C2", compound=aspirin_compound),
                DrugCandidate(candidate_id="C1", compound=aspirin_compound),
                DrugCandidate(candidate_id="C3", compound=aspirin_compound),
            ],
            rankings=[
                CandidateRanking(candidate_id="C1", rank=1, composite_score=0.9),
                CandidateRanking(candidate_id="C2", rank=2, composite_score=0.8),
            ],
        )
        ranked = result.ranked_candidates()
        assert len(ranked) == 3
        assert ranked[0].candidate_id == "C1"
        assert ranked[1].candidate_id == "C2"
        assert ranked[2].candidate_id == "C3"


# =============================================================================
# Tests for utils.py
# =============================================================================

class TestUtils:
    """Test suite for utility functions."""

    def test_is_plausible_smiles_valid(self) -> None:
        """Test that valid SMILES strings pass plausibility check."""
        assert is_plausible_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") is True
        assert is_plausible_smiles("CCO") is True
        assert is_plausible_smiles("c1ccccc1") is True

    def test_is_plausible_smiles_invalid(self) -> None:
        """Test that invalid SMILES strings fail plausibility check."""
        assert is_plausible_smiles("") is False
        # "invalid" passes the character check (letters are allowed)
        # but fails structure check - this is intentional as this is a cheap pre-filter
        assert is_plausible_smiles("C(C") is False  # Unbalanced parentheses
        assert is_plausible_smiles("C[") is False   # Unbalanced bracket
        assert is_plausible_smiles("C)") is False   # Unbalanced parentheses

    def test_validate_non_empty_string_valid(self) -> None:
        """Test that non-empty strings pass validation."""
        result = validate_non_empty_string("test", "field")
        assert result == "test"

    def test_validate_non_empty_string_invalid(self) -> None:
        """Test that empty strings raise InputValidationError."""
        with pytest.raises(InputValidationError, match="must be a non-empty string"):
            validate_non_empty_string("", "field")

        with pytest.raises(InputValidationError, match="must be a string"):
            validate_non_empty_string(123, "field")

    def test_validate_in_range_valid(self) -> None:
        """Test that values within range pass validation."""
        result = validate_in_range(5.0, 0.0, 10.0, "field")
        assert result == 5.0

    def test_validate_in_range_invalid(self) -> None:
        """Test that values outside range raise InputValidationError."""
        with pytest.raises(InputValidationError, match="must be within"):
            validate_in_range(15.0, 0.0, 10.0, "field")

    def test_validate_probability_valid(self) -> None:
        """Test that valid probabilities pass validation."""
        assert validate_probability(0.5, "score") == 0.5
        assert validate_probability(0.0, "score") == 0.0
        assert validate_probability(1.0, "score") == 1.0

    def test_validate_probability_invalid(self) -> None:
        """Test that invalid probabilities raise InputValidationError."""
        with pytest.raises(InputValidationError, match="must be within"):
            validate_probability(1.5, "score")

    def test_validate_positive_integer_valid(self) -> None:
        """Test that positive integers pass validation."""
        assert validate_positive_integer(5, "field") == 5

    def test_validate_positive_integer_invalid(self) -> None:
        """Test that non-positive integers raise InputValidationError."""
        with pytest.raises(InputValidationError, match="must be positive"):
            validate_positive_integer(0, "field")

        with pytest.raises(InputValidationError, match="must be an integer"):
            validate_positive_integer(5.5, "field")

    def test_format_molecular_weight(self) -> None:
        """Test molecular weight formatting."""
        assert format_molecular_weight(180.16) == "180.16 g/mol"
        assert format_molecular_weight(180.16, precision=1) == "180.2 g/mol"

    def test_format_similarity_score(self) -> None:
        """Test similarity score formatting."""
        assert format_similarity_score(0.847) == "0.847"
        assert format_similarity_score(0.847, SimilarityMetric.TANIMOTO) == "0.847 (tanimoto)"

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        assert format_percentage(0.847) == "84.7%"
        assert format_percentage(0.847, precision=0) == "85%"

    def test_clamp(self) -> None:
        """Test clamping function."""
        assert clamp(5.0, 0.0, 1.0) == 1.0
        assert clamp(-1.0, 0.0, 1.0) == 0.0
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_normalize_min_max(self) -> None:
        """Test min-max normalization."""
        assert normalize_min_max(5.0, 0.0, 10.0) == 0.5
        assert normalize_min_max(0.0, 0.0, 10.0) == 0.0
        assert normalize_min_max(10.0, 0.0, 10.0) == 1.0
        assert normalize_min_max(5.0, 5.0, 5.0) == 0.5  # Equal range

    def test_score_to_confidence(self) -> None:
        """Test confidence mapping."""
        assert score_to_confidence(0.9) == PredictionConfidence.VERY_HIGH
        assert score_to_confidence(0.7) == PredictionConfidence.HIGH
        assert score_to_confidence(0.5) == PredictionConfidence.MEDIUM
        assert score_to_confidence(0.3) == PredictionConfidence.LOW

    def test_safe_divide(self) -> None:
        """Test safe division."""
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=1.0) == 1.0

    def test_safe_log(self) -> None:
        """Test safe logarithm."""
        assert safe_log(10.0) > 0
        assert safe_log(-1.0) == 0.0
        assert safe_log(0.0) == 0.0

    def test_safe_mean(self) -> None:
        """Test safe mean calculation."""
        assert safe_mean([1.0, 2.0, 3.0]) == 2.0
        assert safe_mean([]) == 0.0
        assert safe_mean([], default=1.0) == 1.0

    def test_compute_basic_statistics(self) -> None:
        """Test basic statistics computation."""
        stats = compute_basic_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.std_dev == pytest.approx(1.414, rel=0.01)
        assert stats.minimum == 1.0
        assert stats.maximum == 5.0

        stats_empty = compute_basic_statistics([])
        assert stats_empty.count == 0

    def test_generate_unique_id(self) -> None:
        """Test unique ID generation."""
        id1 = generate_unique_id()
        id2 = generate_unique_id()
        assert id1 != id2

        id_with_prefix = generate_unique_id("TEST")
        assert id_with_prefix.startswith("TEST_")
        assert len(id_with_prefix) > 5

    def test_utc_now(self) -> None:
        """Test UTC now function."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo.utcoffset(now) is not None

    def test_deep_merge_dicts(self) -> None:
        """Test deep dictionary merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}
        result = deep_merge_dicts(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 4
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_coerce_bool(self) -> None:
        """Test boolean coercion."""
        assert coerce_bool("true") is True
        assert coerce_bool("True") is True
        assert coerce_bool("1") is True
        assert coerce_bool("yes") is True
        assert coerce_bool("on") is True
        assert coerce_bool("false") is False
        assert coerce_bool("0") is False
        assert coerce_bool("no") is False
        assert coerce_bool("off") is False
        assert coerce_bool(None) is False
        assert coerce_bool(None, default=True) is True

    def test_chunked(self) -> None:
        """Test chunking function."""
        items = list(range(10))
        chunks = list(chunked(items, 3))
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[3] == [9]

    def test_deduplicate_preserving_order(self) -> None:
        """Test deduplication preserving order."""
        items = [1, 2, 1, 3, 2, 4]
        result = deduplicate_preserving_order(items)
        assert result == [1, 2, 3, 4]

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        text = "This is a long text that needs truncation"
        result = truncate_text(text, 20)
        assert len(result) <= 20
        assert result.endswith("...")

        result_no_truncate = truncate_text("Short", 20)
        assert result_no_truncate == "Short"

    @patch("time.perf_counter")
    def test_timed_decorator(self, mock_time: Mock) -> None:
        """Test timing decorator."""
        mock_time.side_effect = [0.0, 1.0]  # Start and end times

        @timed("test_function")
        def test_func() -> str:
            return "result"

        result = test_func()
        assert result == "result"
        assert hasattr(test_func, "last_duration_seconds")
        assert test_func.last_duration_seconds == 1.0  # type: ignore

    def test_retry_decorator_success(self) -> None:
        """Test retry decorator on success."""
        call_count = 0

        @retry(max_attempts=3)
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_retry_then_success(self) -> None:
        """Test retry decorator with retry then success."""
        call_count = 0

        @retry(max_attempts=3, delay_seconds=0.01)
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_exhausted(self) -> None:
        """Test retry decorator when all attempts fail."""
        call_count = 0

        @retry(max_attempts=3, delay_seconds=0.01)
        def test_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError):
            test_func()
        assert call_count == 3


# =============================================================================
# Tests for cache.py
# =============================================================================

class TestCache:
    """Test suite for caching module."""

    def test_cache_config_validation(self) -> None:
        """Test cache configuration validation."""
        config = CacheConfig(max_size=100)
        assert config.max_size == 100

        with pytest.raises(ValueError, match="max_size must be a positive integer"):
            CacheConfig(max_size=0)

        with pytest.raises(ValueError, match="default_ttl_seconds must be positive"):
            CacheConfig(default_ttl_seconds=-1)

    def test_cache_entry_expiration(self) -> None:
        """Test cache entry expiration logic."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=10)
        entry = CacheEntry(
            value="test",
            created_at=now,
            expires_at=expires_at,
            last_accessed_at=now,
        )
        assert entry.is_expired(now) is False
        assert entry.is_expired(expires_at + timedelta(seconds=1)) is True

        # Non-expiring entry
        entry_no_expiry = CacheEntry(
            value="test",
            created_at=now,
            expires_at=None,
            last_accessed_at=now,
        )
        assert entry_no_expiry.is_expired() is False

    def test_cache_basic_operations(self, lru_cache: LRUTTLCache) -> None:
        """Test basic cache operations."""
        # Set and get
        lru_cache.set("key1", "value1")
        found, value = lru_cache.try_get("key1")
        assert found is True
        assert value == "value1"

        # Get with default
        assert lru_cache.get("nonexistent", "default") == "default"

        # Contains
        assert "key1" in lru_cache
        assert "nonexistent" not in lru_cache

        # Size
        assert len(lru_cache) == 1

    def test_cache_eviction(self) -> None:
        """Test LRU cache eviction."""
        config = CacheConfig(max_size=2)
        lru_cache = LRUTTLCache(config=config)

        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")
        lru_cache.set("key3", "value3")

        # key1 should be evicted
        assert "key1" not in lru_cache
        assert "key2" in lru_cache
        assert "key3" in lru_cache
        assert len(lru_cache) == 2

    def test_cache_expiration(self, lru_cache: LRUTTLCache) -> None:
        """Test cache TTL expiration."""
        lru_cache.set("key1", "value1", ttl=0.1)
        assert "key1" in lru_cache

        # Wait for expiration
        import time
        time.sleep(0.2)

        found, value = lru_cache.try_get("key1")
        assert found is False
        assert value is None
        assert "key1" not in lru_cache

    def test_cache_invalidation(self, lru_cache: LRUTTLCache) -> None:
        """Test cache invalidation."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")

        assert lru_cache.invalidate("key1") is True
        assert "key1" not in lru_cache
        assert "key2" in lru_cache
        assert lru_cache.invalidate("nonexistent") is False

    def test_cache_namespace_invalidation(self, lru_cache: LRUTTLCache) -> None:
        """Test namespace invalidation."""
        lru_cache.set("ns1:key1", "value1")
        lru_cache.set("ns1:key2", "value2")
        lru_cache.set("ns2:key3", "value3")

        removed = lru_cache.invalidate_namespace("ns1")
        assert removed == 2
        assert "ns1:key1" not in lru_cache
        assert "ns1:key2" not in lru_cache
        assert "ns2:key3" in lru_cache

    def test_cache_clear(self, lru_cache: LRUTTLCache) -> None:
        """Test cache clearing."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")

        removed = lru_cache.clear()
        assert removed == 2
        assert len(lru_cache) == 0

    def test_cache_stats(self, lru_cache: LRUTTLCache) -> None:
        """Test cache statistics."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", "value2")

        # Hits and misses
        lru_cache.try_get("key1")
        lru_cache.try_get("nonexistent")

        stats = lru_cache.stats()
        assert stats.total_hits == 1
        assert stats.total_misses == 1
        assert stats.current_entry_count == 2
        assert stats.max_size == 100

    def test_cache_key_generation(self) -> None:
        """Test cache key generation."""
        key = make_cache_key("test", "arg1", 123, kwarg="value")
        assert key.startswith("test:")
        assert len(key) > 10

        # Deterministic
        key2 = make_cache_key("test", "arg1", 123, kwarg="value")
        assert key == key2

        # Different args produce different keys
        key3 = make_cache_key("test", "arg2", 123, kwarg="value")
        assert key != key3

    def test_cache_key_generation_with_enum(self) -> None:
        """Test cache key generation with enums."""
        key = make_cache_key("test", SimilarityMetric.TANIMOTO)
        assert key.startswith("test:")

    def test_cache_key_generation_with_dataclass(self, aspirin_compound: Compound) -> None:
        """Test cache key generation with dataclasses."""
        key = make_cache_key("test", aspirin_compound)
        assert key.startswith("test:")

    def test_cache_key_generation_invalid_namespace(self) -> None:
        """Test cache key generation with invalid namespace."""
        with pytest.raises(ValueError, match="namespace must be a non-empty string"):
            make_cache_key("", "arg")

    def test_cache_export_snapshot(self, lru_cache: LRUTTLCache) -> None:
        """Test cache snapshot export."""
        lru_cache.set("key1", "value1")
        lru_cache.set("key2", 123)

        snapshot = lru_cache.export_snapshot()
        assert "key1" in snapshot
        assert "key2" in snapshot
        assert snapshot["key1"]["value"] == "value1"
        assert snapshot["key2"]["value"] == 123

    def test_cache_to_json(self, lru_cache: LRUTTLCache) -> None:
        """Test cache JSON export."""
        lru_cache.set("key1", "value1")
        json_str = lru_cache.to_json()
        assert "key1" in json_str
        assert "value1" in json_str

    def test_default_cache(self) -> None:
        """Test default cache singleton."""
        # Reset default cache
        cache_module._default_cache = None

        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

        # Configure default cache
        config = CacheConfig(max_size=50)
        c3 = cache_module.configure_default_cache(config)
        assert c3.config.max_size == 50

        # Reset
        cache_module.reset_default_cache()
        assert len(c3) == 0

    def test_cached_decorator(self) -> None:
        """Test cached decorator."""
        cache_module._default_cache = None
        default_cache = get_default_cache()

        call_count = 0

        @cached("test_namespace")
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should hit cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1

        # Different args should compute
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2


# =============================================================================
# Tests for molecular_properties.py
# =============================================================================

class TestMolecularProperties:
    """Test suite for molecular properties module."""

    def test_validate_smiles_valid(self) -> None:
        """Test SMILES validation with valid strings."""
        assert validate_smiles("CCO") is True
        assert validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") is True

    def test_validate_smiles_invalid(self) -> None:
        """Test SMILES validation with invalid strings."""
        assert validate_smiles("") is False
        assert validate_smiles("invalid") is False
        assert validate_smiles(123) is False  # type: ignore

    def test_parse_smiles_valid(self) -> None:
        """Test SMILES parsing with valid strings."""
        mol = MolecularPropertyCalculator.parse_smiles("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_parse_smiles_invalid(self) -> None:
        """Test SMILES parsing with invalid strings."""
        with pytest.raises(InvalidSMILESError):
            MolecularPropertyCalculator.parse_smiles("invalid")

        with pytest.raises(InvalidSMILESError):
            MolecularPropertyCalculator.parse_smiles("")

    def test_calculate_lipinski_violations(self) -> None:
        """Test Lipinski violation calculation."""
        # 0 violations
        assert MolecularPropertyCalculator.calculate_lipinski_violations(
            molecular_weight=300.0, logp=2.0, h_bond_donors=2, h_bond_acceptors=4
        ) == 0

        # 4 violations
        assert MolecularPropertyCalculator.calculate_lipinski_violations(
            molecular_weight=600.0, logp=6.0, h_bond_donors=6, h_bond_acceptors=12
        ) == 4

    def test_calculate_properties_valid(self, aspirin_compound: Compound) -> None:
        """Test molecular property calculation with valid SMILES."""
        calculator = MolecularPropertyCalculator()
        props = calculator.calculate(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert props.compound_id == aspirin_compound.compound_id
        assert props.molecular_weight > 0
        assert props.logp is not None
        assert props.lipinski_violations >= 0

    def test_calculate_properties_invalid(self) -> None:
        """Test molecular property calculation with invalid SMILES."""
        calculator = MolecularPropertyCalculator()
        with pytest.raises(InvalidSMILESError):
            calculator.calculate(smiles="invalid", compound_id="C1")

    def test_calculate_extended_descriptors_valid(self, aspirin_compound: Compound) -> None:
        """Test extended descriptor calculation."""
        calculator = MolecularPropertyCalculator()
        ext = calculator.calculate_extended(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert ext.compound_id == aspirin_compound.compound_id
        assert ext.exact_molecular_weight > 0
        assert ext.ring_count >= 0
        assert 0 <= ext.fraction_csp3 <= 1
        assert 0 <= ext.qed_score <= 1

    def test_module_level_functions(self, aspirin_compound: Compound) -> None:
        """Test module-level convenience functions."""
        props = calculate_molecular_properties(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert props.compound_id == aspirin_compound.compound_id

        ext = molecular_properties_module.calculate_extended_descriptors(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert ext.compound_id == aspirin_compound.compound_id


# =============================================================================
# Tests for compound_similarity.py
# =============================================================================

class TestCompoundSimilarity:
    """Test suite for compound similarity module."""

    def test_fingerprint_config_validation(self) -> None:
        """Test fingerprint configuration validation."""
        config = FingerprintConfig()
        assert config.radius == 2
        assert config.n_bits == 2048

        with pytest.raises(ValueError, match="radius cannot be negative"):
            FingerprintConfig(radius=-1)

        with pytest.raises(ValueError, match="n_bits must be a positive integer"):
            FingerprintConfig(n_bits=0)

    def test_generate_fingerprint_valid(self, aspirin_compound: Compound) -> None:
        """Test fingerprint generation with valid SMILES."""
        fp = generate_morgan_fingerprint(aspirin_compound.smiles)
        assert fp is not None
        assert len(fp) == 2048  # Default bit vector length

    def test_generate_fingerprint_invalid(self) -> None:
        """Test fingerprint generation with invalid SMILES."""
        with pytest.raises(InvalidSMILESError):
            generate_morgan_fingerprint("invalid")

    def test_compute_tanimoto_similarity(self) -> None:
        """Test Tanimoto similarity computation."""
        similarity = compound_similarity_module.compute_tanimoto_similarity(
            "CCO",  # Ethanol
            "CCN",  # Ethylamine (similar structure)
        )
        assert 0.0 <= similarity <= 1.0

        # Same compound should have similarity 1.0
        similarity_same = compound_similarity_module.compute_tanimoto_similarity("CCO", "CCO")
        assert similarity_same == 1.0

    def test_find_similar_compounds(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test finding similar compounds."""
        results = find_similar_compounds(
            query_smiles=aspirin_compound.smiles,
            candidates=sample_compounds,
            threshold=0.0,
        )
        # Should find at least aspirin itself
        assert len(results) >= 1

        # With threshold, only highly similar compounds
        results_high = find_similar_compounds(
            query_smiles=aspirin_compound.smiles,
            candidates=sample_compounds,
            threshold=0.9,
        )
        assert len(results_high) <= len(results)

        # Top-K
        results_top = find_similar_compounds(
            query_smiles=aspirin_compound.smiles,
            candidates=sample_compounds,
            threshold=0.0,
            top_k=1,
        )
        assert len(results_top) <= 1

    def test_find_similar_compounds_invalid_query(self, sample_compounds: list[Compound]) -> None:
        """Test finding similar compounds with invalid query."""
        with pytest.raises(InvalidSMILESError):
            find_similar_compounds(
                query_smiles="invalid",
                candidates=sample_compounds,
            )

    def test_find_similar_compounds_skip_invalid(
        self,
        aspirin_compound: Compound,
        invalid_compound: Compound,
    ) -> None:
        """Test finding similar compounds with skip_invalid_candidates."""
        candidates = [aspirin_compound, invalid_compound]
        results = find_similar_compounds(
            query_smiles=aspirin_compound.smiles,
            candidates=candidates,
            threshold=0.0,
            skip_invalid_candidates=True,
        )
        # Should skip invalid compound
        assert len(results) == 1
        assert results[0].compound.compound_id == aspirin_compound.compound_id


# =============================================================================
# Tests for target_prediction.py
# =============================================================================

class TestTargetPrediction:
    """Test suite for target prediction module."""

    def test_rule_based_predictor_initialization(self) -> None:
        """Test rule-based predictor initialization."""
        predictor = RuleBasedTargetPredictor()
        assert len(predictor.rules) > 0
        assert predictor.engine_name == "rule_based_v1"

    def test_rule_based_predictor_predict(self, aspirin_compound: Compound) -> None:
        """Test rule-based predictor predictions."""
        service = TargetPredictionService()
        features = service.extract_features(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )

        predictor = RuleBasedTargetPredictor()
        predictions = predictor.predict(features)

        assert isinstance(predictions, list)
        for pred in predictions:
            assert isinstance(pred, TargetPrediction)
            assert 0.0 <= pred.confidence_score <= 1.0
            assert pred.compound_id == aspirin_compound.compound_id

    def test_predict_targets_valid(self, aspirin_compound: Compound) -> None:
        """Test target prediction with valid input."""
        predictions = predict_targets(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert isinstance(predictions, list)
        if predictions:
            assert 0.0 <= predictions[0].confidence_score <= 1.0

    def test_predict_targets_with_top_k(self, aspirin_compound: Compound) -> None:
        """Test target prediction with top_k limit."""
        predictions = predict_targets(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
            top_k=2,
        )
        assert len(predictions) <= 2

    def test_predict_targets_with_min_confidence(self, aspirin_compound: Compound) -> None:
        """Test target prediction with min_confidence filter."""
        predictions = predict_targets(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
            min_confidence=0.5,
        )
        for pred in predictions:
            assert pred.confidence_score >= 0.5

    def test_predict_targets_invalid_smiles(self) -> None:
        """Test target prediction with invalid SMILES."""
        with pytest.raises(InvalidSMILESError):
            predict_targets(
                smiles="invalid",
                compound_id="C1",
            )

    def test_predict_targets_batch(self, sample_compounds: list[Compound]) -> None:
        """Test batch target prediction."""
        compounds = [(c.smiles, c.compound_id) for c in sample_compounds]
        results = predict_targets_batch(
            compounds=compounds,
            skip_invalid=True,
        )
        assert len(results) == len(sample_compounds)
        for compound_id in results:
            assert isinstance(results[compound_id], list)

    def test_predict_targets_batch_invalid_skip(self, invalid_compound: Compound) -> None:
        """Test batch target prediction with invalid compounds and skip."""
        compounds = [(invalid_compound.smiles, invalid_compound.compound_id)]
        results = predict_targets_batch(
            compounds=compounds,
            skip_invalid=True,
        )
        assert len(results) == 0


# =============================================================================
# Tests for toxicity_prediction.py
# =============================================================================

class TestToxicityPrediction:
    """Test suite for toxicity prediction module."""

    def test_rule_based_predictor_initialization(self) -> None:
        """Test rule-based toxicity predictor initialization."""
        predictor = RuleBasedToxicityPredictor()
        assert predictor.engine_name == "rule_based_v1"
        assert len(predictor.weights) == 4

    def test_rule_based_predictor_predict(self, aspirin_compound: Compound) -> None:
        """Test rule-based toxicity predictor predictions."""
        service = ToxicityPredictionService()
        features = service.extract_features(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )

        predictor = RuleBasedToxicityPredictor()
        report = predictor.predict(features)

        assert isinstance(report, toxicity_prediction_module.ToxicityAssessmentReport)
        assert report.compound_id == aspirin_compound.compound_id
        assert 0.0 <= report.prediction.toxicity_score <= 1.0
        assert isinstance(report.prediction.toxicity_level, ToxicityLevel)
        assert len(report.explanations) > 0

    def test_predict_toxicity_valid(self, aspirin_compound: Compound) -> None:
        """Test toxicity prediction with valid input."""
        prediction = predict_toxicity(
            smiles=aspirin_compound.smiles,
            compound_id=aspirin_compound.compound_id,
        )
        assert isinstance(prediction, ToxicityPrediction)
        assert 0.0 <= prediction.toxicity_score <= 1.0
        assert isinstance(prediction.toxicity_level, ToxicityLevel)

    def test_predict_toxicity_invalid_smiles(self) -> None:
        """Test toxicity prediction with invalid SMILES."""
        with pytest.raises(InvalidSMILESError):
            predict_toxicity(
                smiles="invalid",
                compound_id="C1",
            )

    def test_predict_toxicity_batch(self, sample_compounds: list[Compound]) -> None:
        """Test batch toxicity prediction."""
        compounds = [(c.smiles, c.compound_id) for c in sample_compounds]
        results = predict_toxicity_batch(
            compounds=compounds,
            skip_invalid=True,
        )
        assert len(results) == len(sample_compounds)
        for compound_id in results:
            assert isinstance(results[compound_id], ToxicityPrediction)


# =============================================================================
# Tests for drug_repurposing.py
# =============================================================================

class TestDrugRepurposing:
    """Test suite for drug repurposing module."""

    def test_repurposing_query_validation(self) -> None:
        """Test repurposing query validation."""
        # Valid query with disease name
        query = RepurposingQuery(disease_name="test disease")
        assert query.disease_name == "test disease"

        # Valid query with compound
        with pytest.raises(ValueError):
            RepurposingQuery()

    def test_rule_based_scorer_initialization(self) -> None:
        """Test rule-based scorer initialization."""
        scorer = RuleBasedRepurposingScorer()
        assert scorer.engine_name == "rule_based_v1"
        assert len(scorer.weights) == 4

        # Test custom weights
        custom_weights = {"structural_similarity_score": 0.5, "target_overlap_score": 0.5}
        scorer = RuleBasedRepurposingScorer(weights=custom_weights)
        assert scorer.weights["structural_similarity_score"] == 0.5

    def test_scorer_structural_similarity(
        self,
        aspirin_compound: Compound,
        ibuprofen_compound: Compound,
    ) -> None:
        """Test structural similarity scoring."""
        query = RepurposingQuery(
            query_compound=aspirin_compound,
            disease_name="test",
        )
        scorer = RuleBasedRepurposingScorer()
        explanations = []
        score = scorer._score_structural_similarity(
            query=query,
            candidate=ibuprofen_compound,
            explanations=explanations,
        )
        assert 0.0 <= score <= 1.0

    def test_scorer_target_overlap(
        self,
        aspirin_compound: Compound,
        ibuprofen_compound: Compound,
        drug_target: DrugTarget,
    ) -> None:
        """Test target overlap scoring."""
        query = RepurposingQuery(
            query_compound=aspirin_compound,
            disease_name="test",
        )
        scorer = RuleBasedRepurposingScorer()
        explanations = []

        # Create some target predictions
        target_predictions = [
            TargetPrediction(
                compound_id=aspirin_compound.compound_id,
                target=drug_target,
                confidence_score=0.8,
                confidence_level=PredictionConfidence.HIGH,
            )
        ]

        score, shared = scorer._score_target_overlap(
            query=query,
            candidate_target_predictions=target_predictions,
            query_target_predictions=target_predictions,
            explanations=explanations,
        )
        assert 0.0 <= score <= 1.0

    def test_score_candidate(
        self,
        aspirin_compound: Compound,
        ibuprofen_compound: Compound,
    ) -> None:
        """Test full candidate scoring."""
        query = RepurposingQuery(
            query_compound=aspirin_compound,
            disease_name="test",
        )
        scorer = RuleBasedRepurposingScorer()

        evidence = scorer.score_candidate(
            query=query,
            candidate=ibuprofen_compound,
            candidate_target_predictions=[],
            query_target_predictions=[],
        )
        assert isinstance(evidence, drug_repurposing_module.RepurposingEvidence)
        assert 0.0 <= evidence.composite_score <= 1.0
        assert len(evidence.explanations) > 0

    def test_repurposing_service_find_candidates(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test repurposing service find candidates."""
        service = DrugRepurposingService()

        result = service.find_repurposing_candidates(
            candidates=sample_compounds,
            disease_name="test disease",
            query_compound=aspirin_compound,
        )
        assert isinstance(result, DrugDiscoveryResult)
        assert len(result.candidates) <= len(sample_compounds)

    def test_find_repurposing_candidates_module(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test module-level repurposing function."""
        result = find_repurposing_candidates(
            candidates=sample_compounds,
            disease_name="test disease",
            query_compound=aspirin_compound,
        )
        assert isinstance(result, DrugDiscoveryResult)

    def test_find_repurposing_candidates_with_top_k(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test repurposing with top_k limit."""
        result = find_repurposing_candidates(
            candidates=sample_compounds,
            disease_name="test disease",
            query_compound=aspirin_compound,
            top_k=2,
        )
        assert len(result.candidates) <= 2


# =============================================================================
# Tests for candidate_ranking.py
# =============================================================================

class TestCandidateRanking:
    """Test suite for candidate ranking module."""

    def test_ranking_criterion_validation(self) -> None:
        """Test ranking criterion validation."""
        criterion = candidate_ranking_module.RankingCriterion(
            name="test",
            weight=1.0,
            higher_is_better=True,
        )
        assert criterion.name == "test"

        with pytest.raises(candidate_ranking_module.RankingConfigurationError):
            candidate_ranking_module.RankingCriterion(name="", weight=1.0)

        with pytest.raises(candidate_ranking_module.RankingConfigurationError):
            candidate_ranking_module.RankingCriterion(name="test", weight=-1.0)

    def test_ranking_config_validation(self) -> None:
        """Test ranking configuration validation."""
        config = CandidateRankingConfig()
        assert len(config.criteria) > 0

        # Empty criteria
        with pytest.raises(candidate_ranking_module.RankingConfigurationError):
            CandidateRankingConfig(criteria=[])

    def test_ranking_config_with_weights(self) -> None:
        """Test ranking configuration with custom weights."""
        config = CandidateRankingConfig()
        new_config = config.with_weights({"structural_similarity": 2.0})
        assert new_config.weights.get("structural_similarity", 0) == 2.0

    def test_score_normalizer_min_max(self) -> None:
        """Test min-max normalization."""
        values = {"a": 0.0, "b": 0.5, "c": 1.0}
        normalized = candidate_ranking_module.ScoreNormalizer.normalize(
            values=values,
            strategy=candidate_ranking_module.NormalizationStrategy.MIN_MAX,
            higher_is_better=True,
        )
        assert normalized["a"] == 0.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 1.0

        # All equal values
        values_equal = {"a": 0.5, "b": 0.5, "c": 0.5}
        normalized = candidate_ranking_module.ScoreNormalizer.normalize(
            values=values_equal,
            strategy=candidate_ranking_module.NormalizationStrategy.MIN_MAX,
            higher_is_better=True,
        )
        assert normalized["a"] == 1.0

    def test_score_normalizer_clamp(self) -> None:
        """Test clamp normalization."""
        values = {"a": -0.5, "b": 0.5, "c": 1.5}
        normalized = candidate_ranking_module.ScoreNormalizer.normalize(
            values=values,
            strategy=candidate_ranking_module.NormalizationStrategy.CLAMP,
            higher_is_better=True,
        )
        assert normalized["a"] == 0.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 1.0

    def test_score_normalizer_higher_is_better_false(self) -> None:
        """Test normalization with higher_is_better=False."""
        values = {"a": 0.0, "b": 0.5, "c": 1.0}
        normalized = candidate_ranking_module.ScoreNormalizer.normalize(
            values=values,
            strategy=candidate_ranking_module.NormalizationStrategy.MIN_MAX,
            higher_is_better=False,
        )
        assert normalized["a"] == 1.0
        assert normalized["b"] == 0.5
        assert normalized["c"] == 0.0

    def test_weighted_sum_ranking_strategy(self) -> None:
        """Test weighted sum ranking strategy."""
        strategy = WeightedSumRankingStrategy()
        normalized_scores = {"criteria1": 0.8, "criteria2": 0.6}
        weights = {"criteria1": 1.0, "criteria2": 1.0}
        result = strategy.combine(normalized_scores, weights)
        assert result == 0.7

    def test_weighted_geometric_mean_ranking_strategy(self) -> None:
        """Test weighted geometric mean ranking strategy."""
        strategy = candidate_ranking_module.WeightedGeometricMeanRankingStrategy()
        normalized_scores = {"criteria1": 0.8, "criteria2": 0.6}
        weights = {"criteria1": 1.0, "criteria2": 1.0}
        result = strategy.combine(normalized_scores, weights)
        assert 0.0 <= result <= 1.0

    def test_ranking_filter_satisfied(self) -> None:
        """Test ranking filter satisfaction."""
        # Create a bundle
        bundle = CandidateEvidenceBundle(
            compound=Compound(compound_id="C1", name="Test", smiles="C"),
        )
        filter_criteria = RankingFilter(
            min_composite_score=0.5,
        )
        assert filter_criteria.is_satisfied_by(bundle, 0.6, PredictionConfidence.MEDIUM) is True
        assert filter_criteria.is_satisfied_by(bundle, 0.4, PredictionConfidence.MEDIUM) is False

    def test_ranking_filter_toxicity(self) -> None:
        """Test ranking filter with toxicity level."""
        bundle = CandidateEvidenceBundle(
            compound=Compound(compound_id="C1", name="Test", smiles="C"),
            toxicity_prediction=ToxicityPrediction(
                compound_id="C1",
                toxicity_level=ToxicityLevel.HIGH,
                toxicity_score=0.8,
            ),
        )
        filter_criteria = RankingFilter(
            max_toxicity_level=ToxicityLevel.MODERATE,
        )
        assert filter_criteria.is_satisfied_by(bundle, 0.5, PredictionConfidence.MEDIUM) is False

        filter_criteria_low = RankingFilter(
            max_toxicity_level=ToxicityLevel.HIGH,
        )
        assert filter_criteria_low.is_satisfied_by(bundle, 0.5, PredictionConfidence.MEDIUM) is True

    def test_candidate_ranking_engine_rank(
        self,
        sample_compounds: list[Compound],
        molecular_properties: MolecularProperties,
    ) -> None:
        """Test candidate ranking engine."""
        engine = CandidateRankingEngine()

        # Create evidence bundles
        bundles = []
        for compound in sample_compounds:
            bundle = CandidateEvidenceBundle(
                compound=compound,
                molecular_properties=molecular_properties,
            )
            bundles.append(bundle)

        result = engine.rank(bundles=bundles)
        assert isinstance(result, DrugDiscoveryResult)
        assert len(result.candidates) == len(sample_compounds)
        assert len(result.rankings) == len(sample_compounds)

        # Check rankings are sorted
        scores = [r.composite_score for r in result.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_candidate_ranking_engine_with_top_k(
        self,
        sample_compounds: list[Compound],
    ) -> None:
        """Test candidate ranking engine with top_k."""
        engine = CandidateRankingEngine()

        bundles = []
        for compound in sample_compounds:
            bundle = CandidateEvidenceBundle(
                compound=compound,
            )
            bundles.append(bundle)

        result = engine.rank(bundles=bundles, top_k=2)
        assert len(result.candidates) <= 2

    def test_rank_candidates_module(
        self,
        sample_compounds: list[Compound],
    ) -> None:
        """Test module-level rank_candidates function."""
        bundles = []
        for compound in sample_compounds:
            bundle = CandidateEvidenceBundle(
                compound=compound,
            )
            bundles.append(bundle)

        result = rank_candidates(bundles=bundles)
        assert isinstance(result, DrugDiscoveryResult)

    def test_ranking_statistics_computation(
        self,
        sample_compounds: list[Compound],
    ) -> None:
        """Test ranking statistics computation."""
        rankings = [
            CandidateRanking(
                candidate_id=f"C{i}",
                rank=i+1,
                composite_score=1.0 - i * 0.1,
                criteria_scores={"test": 0.5},
                notes="Confidence: HIGH. Test note.",
            )
            for i in range(3)
        ]

        stats = candidate_ranking_module.CandidateRankingEngine.compute_statistics(rankings)
        assert stats.candidate_count == 3
        # Scores: 1.0, 0.9, 0.8 -> mean = 0.9
        assert stats.mean_composite_score == 0.9
        assert stats.max_composite_score == 1.0
        assert stats.min_composite_score == 0.8
        assert "high" in stats.confidence_distribution


# =============================================================================
# Tests for pipeline.py
# =============================================================================

class TestPipeline:
    """Test suite for the Drug Discovery pipeline."""

    def test_pipeline_config_validation(self) -> None:
        """Test pipeline configuration validation."""
        config = PipelineConfig()
        assert config.top_k is None
        assert config.enable_cache is True

        config_with_top_k = PipelineConfig(top_k=10)
        assert config_with_top_k.top_k == 10

        with pytest.raises(ValueError):
            PipelineConfig(max_compound_errors=0)

    def test_pipeline_initialization(self, pipeline_config: PipelineConfig) -> None:
        """Test pipeline initialization."""
        pipeline_instance = DrugDiscoveryPipeline(config=pipeline_config)
        assert pipeline_instance.config is pipeline_config

    def test_pipeline_run_basic(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test basic pipeline execution."""
        # Create pipeline with caching disabled for deterministic tests
        config = pipeline_config
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=sample_compounds,
            query_smiles=aspirin_compound.smiles,
            disease_name="test disease",
        )

        assert isinstance(result, DrugDiscoveryResult)
        assert result.run_id.startswith("DD_")
        assert len(result.candidates) > 0
        assert result.completed_at is not None
        assert result.completed_at >= result.created_at

    def test_pipeline_run_with_single_compound(
        self,
        aspirin_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with a single compound."""
        config = pipeline_config
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )

        assert isinstance(result, DrugDiscoveryResult)
        assert len(result.candidates) == 1

    def test_pipeline_run_with_invalid_compound_skip(
        self,
        aspirin_compound: Compound,
        invalid_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with invalid compounds and skip enabled."""
        config = pipeline_config
        config.enable_cache = False
        config.skip_invalid_compounds = True

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound, invalid_compound],
            query_smiles=aspirin_compound.smiles,
    )

        # Should still produce results
        assert len(result.candidates) > 0
        # At least the valid compound should be present
        valid_candidates = [c for c in result.candidates if c.compound.compound_id == "C_ASPIRIN"]
        assert len(valid_candidates) > 0

    def test_pipeline_run_with_invalid_compound_no_skip(
        self,
        aspirin_compound: Compound,
        invalid_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with invalid compounds and skip disabled."""
        config = pipeline_config
        config.enable_cache = False
        config.skip_invalid_compounds = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)

        result = pipeline_instance.run(
            compounds=[aspirin_compound, invalid_compound],
            query_smiles=aspirin_compound.smiles,
        )
    
        # The pipeline completed but both compounds are in results
        # Check that the invalid compound is still in the results
        invalid_candidates = [c for c in result.candidates if c.compound.compound_id == "C_INVALID"]
        assert len(invalid_candidates) == 1
        # Check that the valid compound is also present
        valid_candidates = [c for c in result.candidates if c.compound.compound_id == "C_ASPIRIN"]
        assert len(valid_candidates) == 1

    def test_pipeline_run_with_disabled_stages(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with various stages disabled."""
        config = pipeline_config
        config.enable_cache = False
        config.enable_similarity = False
        config.enable_target_prediction = False
        config.enable_toxicity_prediction = False
        config.enable_repurposing = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )

        assert isinstance(result, DrugDiscoveryResult)
        # Should still produce results (just molecular properties and ranking)
        assert len(result.candidates) > 0

    def test_pipeline_metadata_collection(
        self,
        aspirin_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline metadata collection."""
        config = pipeline_config
        config.enable_cache = False
        config.collect_timing = True

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )

        assert "total_duration_seconds" in result.metadata
        assert "stage_timing" in result.metadata
        assert "stage_results" in result.metadata

    def test_pipeline_config_override(
        self,
        aspirin_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with configuration overrides."""
        config = pipeline_config
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
            config_override={"top_k": 5},
        )

        assert result.metadata["pipeline_config"]["top_k"] == 5

    def test_pipeline_should_abort_on_stage_failure(self) -> None:
        """Test pipeline abort logic on stage failure."""
        config = PipelineConfig()
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)

        # Critical stages should abort
        assert pipeline_instance._should_abort_on_stage_failure("input_validation") is True
        assert pipeline_instance._should_abort_on_stage_failure("ranking") is True

        # Non-critical stages should not abort
        assert pipeline_instance._should_abort_on_stage_failure("similarity") is False
        assert pipeline_instance._should_abort_on_stage_failure("target_prediction") is False

    def test_pipeline_error_limit(
        self,
        aspirin_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline error limit behavior."""
        config = pipeline_config
        config.enable_cache = False
        config.max_compound_errors = 1

        pipeline_instance = DrugDiscoveryPipeline(config=config)

        # Should complete without issues
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )
        assert len(result.candidates) > 0


# =============================================================================
# Tests for module-level convenience functions
# =============================================================================

class TestModuleFunctions:
    """Test suite for module-level convenience functions."""

    def test_run_drug_discovery(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test the run_drug_discovery convenience function."""
        result = run_drug_discovery(
            compounds=sample_compounds,
            query_smiles=aspirin_compound.smiles,
            disease_name="test disease",
        )
        assert isinstance(result, DrugDiscoveryResult)
        assert len(result.candidates) > 0

    def test_run_drug_discovery_with_config(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test run_drug_discovery with custom config."""
        config = pipeline_config
        config.enable_cache = False

        result = run_drug_discovery(
            compounds=sample_compounds,
            query_smiles=aspirin_compound.smiles,
            disease_name="test disease",
            config=config,
        )
        assert isinstance(result, DrugDiscoveryResult)

    def test_run_drug_discovery_with_kwargs(
        self,
        aspirin_compound: Compound,
        sample_compounds: list[Compound],
    ) -> None:
        """Test run_drug_discovery with kwargs overrides."""
        result = run_drug_discovery(
            compounds=sample_compounds,
            query_smiles=aspirin_compound.smiles,
            disease_name="test disease",
            top_k=5,
            enable_cache=False,
        )
        assert isinstance(result, DrugDiscoveryResult)
        assert result.metadata["pipeline_config"]["top_k"] == 5


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete Drug Discovery pipeline."""

    def test_full_pipeline_integration(
        self,
        aspirin_compound: Compound,
        ibuprofen_compound: Compound,
        paracetamol_compound: Compound,
    ) -> None:
        """Test the full pipeline end-to-end with real computations."""
        compounds = [aspirin_compound, ibuprofen_compound, paracetamol_compound]

        config = PipelineConfig(
            top_k=10,
            enable_cache=False,
            enable_similarity=True,
            enable_target_prediction=True,
            enable_toxicity_prediction=True,
            enable_repurposing=True,
            collect_timing=True,
        )
        config.discovery_config.similarity_threshold = 0.3

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=compounds,
            query_smiles=aspirin_compound.smiles,
            disease_name="inflammation",
        )

        # Check result structure
        assert isinstance(result, DrugDiscoveryResult)
        assert result.run_id.startswith("DD_")
        assert len(result.candidates) > 0
        assert len(result.rankings) == len(result.candidates)

        # Check stage progression
        assert result.stage == DiscoveryStage.CANDIDATE_RANKING or result.stage == DiscoveryStage.COMPLETED

        # Check metadata
        assert "total_duration_seconds" in result.metadata
        assert "stage_timing" in result.metadata

        # Check ranking scores
        for ranking in result.rankings:
            assert 0.0 <= ranking.composite_score <= 1.0
            assert ranking.rank >= 1

        # Check candidates have properties
        for candidate in result.candidates:
            assert candidate.compound is not None

    def test_pipeline_caching_integration(
        self,
        aspirin_compound: Compound,
    ) -> None:
        """Test pipeline with caching enabled."""
        config = PipelineConfig(
            top_k=5,
            enable_cache=True,
            enable_similarity=True,
            enable_target_prediction=True,
            enable_toxicity_prediction=True,
            enable_repurposing=True,
        )

        pipeline_instance = DrugDiscoveryPipeline(config=config)

        # First run
        result1 = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )

        # Second run should use cache
        result2 = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=aspirin_compound.smiles,
        )

        # Both should produce valid results
        assert len(result1.candidates) == len(result2.candidates)

    def test_pipeline_with_empty_compounds(
        self,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with empty compound list."""
        config = pipeline_config
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
    
        result = pipeline_instance.run(
        compounds=[],
        query_smiles="CCO",
        )
        # The pipeline should have warnings about no valid compounds
        assert len(result.warnings) > 0
        # Check that the input_validation stage failed
        stage_result = result.metadata.get("stage_results", {}).get("input_validation")
        if stage_result:
            assert stage_result["success"] is False

    def test_pipeline_with_no_query_smiles(
        self,
        aspirin_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline without query SMILES."""
        config = pipeline_config
        config.enable_cache = False

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound],
            query_smiles=None,
        )

        # Should still work
        assert isinstance(result, DrugDiscoveryResult)
        assert len(result.candidates) > 0


# =============================================================================
# Performance tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_cache_performance(self, lru_cache: LRUTTLCache) -> None:
        """Test cache performance with many operations."""
        # Add many items
        for i in range(50):
            lru_cache.set(f"key{i}", f"value{i}")

        assert len(lru_cache) == 50

        # Hit all items
        for i in range(50):
            found, value = lru_cache.try_get(f"key{i}")
            assert found is True
            assert value == f"value{i}"

        # Add more items to trigger eviction
        for i in range(50, 75):
            lru_cache.set(f"key{i}", f"value{i}")

        # Should still be at max size
        assert len(lru_cache) <= lru_cache.config.max_size

    def test_ranking_performance(
        self,
        sample_compounds: list[Compound],
    ) -> None:
        """Test ranking performance with many candidates."""
        # Create many candidate bundles
        bundles = []
        for i in range(20):
            compound = Compound(
                compound_id=f"C{i}",
                name=f"Compound {i}",
                smiles="CCO",
            )
            bundle = CandidateEvidenceBundle(
                compound=compound,
                molecular_properties=MolecularProperties(
                    compound_id=f"C{i}",
                    molecular_weight=100.0 + i,
                    logp=1.0 + i * 0.1,
                    tpsa=50.0,
                    h_bond_donors=1,
                    h_bond_acceptors=2,
                    rotatable_bonds=3,
                    aromatic_rings=1,
                    heavy_atom_count=10,
                    lipinski_violations=0,
                    qed_score=0.8 - i * 0.01,
                ),
            )
            bundles.append(bundle)

        engine = CandidateRankingEngine()
        result = engine.rank(bundles=bundles)

        assert len(result.candidates) == len(bundles)
        # Check ranking is sorted
        scores = [r.composite_score for r in result.rankings]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_cache_edge_cases(self) -> None:
        """Test cache edge cases."""
        config = CacheConfig(max_size=10)
        lru_cache = LRUTTLCache(config=config)

        # Test None value
        lru_cache.set("none_key", None)
        found, value = lru_cache.try_get("none_key")
        assert found is True
        assert value is None

        # Test empty string key
        lru_cache.set("", "empty_key")
        found, value = lru_cache.try_get("")
        assert found is True
        assert value == "empty_key"

        # Test very large key
        large_key = "x" * 10000
        lru_cache.set(large_key, "large_value")
        found, value = lru_cache.try_get(large_key)
        assert found is True
        assert value == "large_value"

    def test_target_prediction_edge_cases(self) -> None:
        """Test target prediction edge cases."""
        # Empty SMILES
        with pytest.raises(InvalidSMILESError):
            predict_targets(smiles="", compound_id="C1")

        # Compound with no matching rules
        predictions = predict_targets(
            smiles="C1=CC=CC=C1",  # Simple benzene
            compound_id="C_NO_MATCH",
        )
        assert isinstance(predictions, list)
        # May be empty or have some predictions

    def test_toxicity_prediction_edge_cases(self) -> None:
        """Test toxicity prediction edge cases."""
        # Very simple molecule
        prediction = predict_toxicity(
            smiles="C",
            compound_id="C_METHANE",
        )
        assert isinstance(prediction, ToxicityPrediction)
        assert 0.0 <= prediction.toxicity_score <= 1.0

        # Very complex molecule (tannic acid-like)
        prediction = predict_toxicity(
            smiles="CC1=C(C(=C(C(=C1O)O)O)O)C2=CC(=C(C=C2O)O)O",
            compound_id="C_COMPLEX",
        )
        assert isinstance(prediction, ToxicityPrediction)

    def test_ranking_edge_cases(self) -> None:
        """Test ranking edge cases."""
        engine = CandidateRankingEngine()

        # Empty bundles
        result = engine.rank(bundles=[])
        assert len(result.candidates) == 0
        assert len(result.rankings) == 0

        # Single bundle with no criteria
        compound = Compound(compound_id="C1", name="Test", smiles="C")
        bundle = CandidateEvidenceBundle(compound=compound)
        result = engine.rank(bundles=[bundle])
        assert len(result.candidates) == 1
        assert result.rankings[0].rank == 1

    def test_pipeline_with_mixed_valid_invalid(
        self,
        aspirin_compound: Compound,
        invalid_compound: Compound,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test pipeline with a mix of valid and invalid compounds."""
        config = pipeline_config
        config.enable_cache = False
        config.skip_invalid_compounds = True

        pipeline_instance = DrugDiscoveryPipeline(config=config)
        result = pipeline_instance.run(
            compounds=[aspirin_compound, invalid_compound],
            query_smiles=aspirin_compound.smiles,
        )

        # Valid compound should be processed
        assert len(result.candidates) > 0
        valid_candidates = [c for c in result.candidates if c.compound.compound_id == "C_ASPIRIN"]
        assert len(valid_candidates) > 0