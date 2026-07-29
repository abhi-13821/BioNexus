"""
drug_discovery/candidate_ranking.py

Multi-source candidate ranking for the BioNexus Drug Discovery module.

This module combines evidence produced by the other Drug Discovery
modules -- ``molecular_properties``, ``compound_similarity``,
``target_prediction``, ``toxicity_prediction``, and
``drug_repurposing`` -- into a single, configurable ranking of candidate
compounds.

Responsibilities:
    - Aggregate heterogeneous per-candidate evidence into a common
      ``CandidateEvidenceBundle`` representation.
    - Extract raw scoring criteria from that evidence (structural
      similarity, predicted target confidence, toxicity risk,
      drug-likeness, Lipinski compliance, repurposing evidence, and any
      caller-supplied custom criteria).
    - Normalize heterogeneous raw values onto a common [0.0, 1.0] scale.
    - Combine normalized criteria into a composite score via a pluggable
      ``RankingStrategy`` (weighted sum, weighted geometric mean, or a
      custom future strategy), supporting true multi-objective scoring.
    - Assign a qualitative confidence level to each ranking.
    - Generate human-readable explanations for each candidate's ranking.
    - Support filtering, top-k selection, and deterministic tie-breaking.
    - Compute aggregate ranking statistics and identify the
      Pareto-optimal (non-dominated) subset of candidates.
    - Produce a human-readable ranking report.

Results are returned using the ``DrugCandidate``, ``CandidateRanking``,
and ``DrugDiscoveryResult`` dataclasses defined in
``drug_discovery.models``, so this module integrates directly with
``pipeline.py`` and the rest of the Drug Discovery module.

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
import math
import statistics
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from drug_discovery.models import (
    CandidateRanking,
    CandidateStatus,
    Compound,
    DiscoveryStage,
    DrugCandidate,
    DrugDiscoveryConfig,
    DrugDiscoveryResult,
    MolecularProperties,
    PredictionConfidence,
    SimilarCompound,
    TargetPrediction,
    ToxicityLevel,
    ToxicityPrediction,
)

__all__ = [
    "CandidateRankingError",
    "RankingConfigurationError",
    "NormalizationStrategy",
    "RankingCriterion",
    "CandidateEvidenceBundle",
    "CandidateRankingConfig",
    "RankingFilter",
    "CandidateFilterPredicate",
    "RankingStatistics",
    "RankingMetadata",
    "RankingStrategy",
    "WeightedSumRankingStrategy",
    "WeightedGeometricMeanRankingStrategy",
    "ScoreNormalizer",
    "CandidateRankingEngine",
    "rank_candidates",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CandidateRankingError(RuntimeError):
    """
    Raised when candidate ranking fails unexpectedly for an otherwise
    valid set of evidence bundles.
    """


class RankingConfigurationError(ValueError):
    """
    Raised when a ``CandidateRankingConfig`` or related configuration is
    invalid (e.g., no criteria defined, non-positive total weight, or an
    unresolvable criterion reference).
    """


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NormalizationStrategy(str, Enum):
    """Strategies for normalizing raw criterion values onto [0.0, 1.0]."""

    MIN_MAX = "min_max"
    """Rescale each criterion's values linearly to span [0.0, 1.0]."""

    CLAMP = "clamp"
    """Assume values are already approximately unit-scaled; clip to
    [0.0, 1.0] without rescaling."""

    Z_SCORE = "z_score"
    """Standardize to a z-score, then squash to [0.0, 1.0] via a logistic
    transform. Useful when raw distributions are roughly normal and
    outliers should be compressed rather than stretching the whole
    scale."""


# ---------------------------------------------------------------------------
# Ranking criteria and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingCriterion:
    """
    A single scoring criterion contributing to a candidate's composite
    ranking score.

    Attributes:
        name: Unique identifier for the criterion (e.g.,
            "structural_similarity"). Must match a key produced by the
            evidence extraction step.
        weight: Relative importance of this criterion. Weights across a
            ``CandidateRankingConfig`` are normalized automatically and
            need not sum to 1.0.
        higher_is_better: Whether higher raw values represent more
            desirable outcomes. Set to False for risk-like metrics (e.g.,
            toxicity score, Lipinski violation count).
        description: Human-readable description of what this criterion
            measures, used in generated explanations and reports.
    """

    name: str
    weight: float
    higher_is_better: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the criterion's name and weight."""
        if not self.name.strip():
            raise RankingConfigurationError("Criterion name must be non-empty.")
        if self.weight < 0:
            raise RankingConfigurationError(
                f"Weight for criterion '{self.name}' cannot be negative."
            )


def _default_ranking_criteria() -> list[RankingCriterion]:
    """
    Build the default set of ranking criteria spanning all upstream Drug
    Discovery modules.

    Returns:
        A list of ``RankingCriterion`` instances with equal default
        weighting.
    """
    return [
        RankingCriterion(
            name="structural_similarity",
            weight=1.0,
            higher_is_better=True,
            description=(
                "Structural (Tanimoto) similarity to a reference "
                "compound, from compound_similarity."
            ),
        ),
        RankingCriterion(
            name="target_confidence",
            weight=1.0,
            higher_is_better=True,
            description=(
                "Confidence of the strongest predicted biological "
                "target interaction, from target_prediction."
            ),
        ),
        RankingCriterion(
            name="toxicity_risk",
            weight=1.0,
            higher_is_better=False,
            description=(
                "Predicted overall toxicity risk score, from "
                "toxicity_prediction (lower is safer)."
            ),
        ),
        RankingCriterion(
            name="drug_likeness",
            weight=1.0,
            higher_is_better=True,
            description=(
                "QED drug-likeness score, from molecular_properties."
            ),
        ),
        RankingCriterion(
            name="lipinski_violations",
            weight=0.5,
            higher_is_better=False,
            description=(
                "Number of Lipinski Rule of Five violations, from "
                "molecular_properties (lower is better)."
            ),
        ),
        RankingCriterion(
            name="repurposing_evidence",
            weight=1.0,
            higher_is_better=True,
            description=(
                "Composite repurposing evidence score, from "
                "drug_repurposing."
            ),
        ),
    ]


@dataclass
class CandidateRankingConfig:
    """
    Configuration governing how candidates are scored and ranked.

    Attributes:
        criteria: The ranking criteria to evaluate. Each criterion's
            ``name`` must correspond to a key that
            ``CandidateRankingEngine`` can extract from a
            ``CandidateEvidenceBundle`` (either a built-in criterion or a
            key present in the bundle's ``custom_scores`` mapping).
        normalization_strategy: The strategy used to normalize raw
            criterion values across the candidate pool before combining
            them.
    """

    criteria: list[RankingCriterion] = field(default_factory=_default_ranking_criteria)
    normalization_strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX

    def __post_init__(self) -> None:
        """Validate that at least one criterion with positive total
        weight is configured."""
        if not self.criteria:
            raise RankingConfigurationError(
                "At least one ranking criterion must be configured."
            )
        if sum(criterion.weight for criterion in self.criteria) <= 0:
            raise RankingConfigurationError(
                "Total criterion weight must be greater than zero."
            )

    @property
    def weights(self) -> dict[str, float]:
        """
        Return the (unnormalized) weight for each configured criterion.

        Returns:
            A mapping of criterion name to weight.
        """
        return {criterion.name: criterion.weight for criterion in self.criteria}

    @property
    def directions(self) -> dict[str, bool]:
        """
        Return the ``higher_is_better`` flag for each configured
        criterion.

        Returns:
            A mapping of criterion name to its direction flag.
        """
        return {criterion.name: criterion.higher_is_better for criterion in self.criteria}

    def with_weights(self, weights: dict[str, float]) -> "CandidateRankingConfig":
        """
        Return a new configuration with updated weights for the given
        criteria, leaving all other criteria and settings unchanged.

        Args:
            weights: Mapping of criterion name to its new weight. Names
                not already present in ``self.criteria`` are ignored with
                a warning.

        Returns:
            A new ``CandidateRankingConfig`` instance reflecting the
            updated weights.
        """
        known_names = {criterion.name for criterion in self.criteria}
        unknown = set(weights) - known_names
        if unknown:
            logger.warning(
                "Ignoring weights for unknown ranking criteria: %s",
                ", ".join(sorted(unknown)),
            )
        updated_criteria = [
            replace(criterion, weight=weights.get(criterion.name, criterion.weight))
            for criterion in self.criteria
        ]
        return replace(self, criteria=updated_criteria)

    @classmethod
    def default(cls) -> "CandidateRankingConfig":
        """
        Build the default ranking configuration covering all upstream
        Drug Discovery modules with equal base weighting.

        Returns:
            A new ``CandidateRankingConfig`` instance.
        """
        return cls()

    @classmethod
    def from_discovery_config(
        cls, discovery_config: DrugDiscoveryConfig
    ) -> "CandidateRankingConfig":
        """
        Build a ranking configuration from a pipeline-level
        ``DrugDiscoveryConfig``, applying any custom ranking weights it
        specifies on top of the default criteria set.

        Args:
            discovery_config: The pipeline's ``DrugDiscoveryConfig``,
                whose ``ranking_weights`` mapping is used to override
                default criterion weights.

        Returns:
            A new ``CandidateRankingConfig`` instance.
        """
        config = cls.default()
        if discovery_config.ranking_weights:
            config = config.with_weights(discovery_config.ranking_weights)
        return config


# ---------------------------------------------------------------------------
# Evidence bundle
# ---------------------------------------------------------------------------


@dataclass
class CandidateEvidenceBundle:
    """
    Aggregated per-candidate evidence collected from the other Drug
    Discovery modules, serving as the input unit for
    ``CandidateRankingEngine``.

    Attributes:
        compound: The candidate ``Compound``.
        molecular_properties: Computed physicochemical properties, from
            ``molecular_properties``.
        similar_compound: Structural similarity evidence relative to a
            reference/query compound, from ``compound_similarity``.
        target_predictions: Predicted biological target interactions,
            from ``target_prediction``.
        toxicity_prediction: Predicted toxicity profile, from
            ``toxicity_prediction``.
        repurposing_score: Composite repurposing evidence score, in
            [0.0, 1.0], from ``drug_repurposing``.
        repurposing_confidence: Confidence bucket associated with
            ``repurposing_score``, from ``drug_repurposing``.
        custom_scores: Additional caller-supplied raw scores, in
            [0.0, 1.0], keyed by criterion name. Enables extending the
            ranking with new evidence sources without modifying this
            dataclass -- simply add a matching ``RankingCriterion`` to
            the active ``CandidateRankingConfig``.
    """

    compound: Compound
    molecular_properties: MolecularProperties | None = None
    similar_compound: SimilarCompound | None = None
    target_predictions: list[TargetPrediction] = field(default_factory=list)
    toxicity_prediction: ToxicityPrediction | None = None
    repurposing_score: float | None = None
    repurposing_confidence: PredictionConfidence | None = None
    custom_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optional numeric fields."""
        if self.repurposing_score is not None and not 0.0 <= self.repurposing_score <= 1.0:
            raise ValueError("repurposing_score must be within [0.0, 1.0].")
        for name, value in self.custom_scores.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"custom_scores['{name}'] must be within [0.0, 1.0]."
                )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

CandidateFilterPredicate = Callable[
    [CandidateEvidenceBundle, float, PredictionConfidence], bool
]
"""A custom filter predicate receiving a candidate's evidence bundle, its
computed composite score, and its computed confidence level, and
returning True if the candidate should be retained."""


@dataclass
class RankingFilter:
    """
    Filtering criteria applied to candidates after scoring, before final
    ranking and top-k selection.

    Attributes:
        min_composite_score: Minimum composite score, in [0.0, 1.0],
            required for a candidate to be retained.
        max_toxicity_level: Maximum acceptable ``ToxicityLevel``;
            candidates with a strictly higher-risk level are excluded.
            Ordering is LOW < MODERATE < HIGH < SEVERE < UNKNOWN, where
            UNKNOWN is treated as the most restrictive (always excluded
            unless explicitly allowed by setting this to
            ``ToxicityLevel.UNKNOWN``).
        require_lipinski_compliant: If True, only candidates with zero
            Lipinski Rule of Five violations are retained.
        min_confidence: Minimum required qualitative confidence bucket
            for the composite ranking.
        custom_predicates: Additional custom filter predicates; a
            candidate must satisfy all of them to be retained.
    """

    min_composite_score: float | None = None
    max_toxicity_level: ToxicityLevel | None = None
    require_lipinski_compliant: bool = False
    min_confidence: PredictionConfidence | None = None
    custom_predicates: list[CandidateFilterPredicate] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the minimum composite score threshold, if provided."""
        if self.min_composite_score is not None and not (
            0.0 <= self.min_composite_score <= 1.0
        ):
            raise ValueError("min_composite_score must be within [0.0, 1.0].")

    _TOXICITY_ORDER: dict[ToxicityLevel, int] = field(
        default_factory=lambda: {
            ToxicityLevel.LOW: 0,
            ToxicityLevel.MODERATE: 1,
            ToxicityLevel.HIGH: 2,
            ToxicityLevel.SEVERE: 3,
            ToxicityLevel.UNKNOWN: 4,
        },
        init=False,
        repr=False,
        compare=False,
    )

    _CONFIDENCE_ORDER: dict[PredictionConfidence, int] = field(
        default_factory=lambda: {
            PredictionConfidence.LOW: 0,
            PredictionConfidence.MEDIUM: 1,
            PredictionConfidence.HIGH: 2,
            PredictionConfidence.VERY_HIGH: 3,
        },
        init=False,
        repr=False,
        compare=False,
    )

    def is_satisfied_by(
        self,
        bundle: CandidateEvidenceBundle,
        composite_score: float,
        confidence: PredictionConfidence,
    ) -> bool:
        """
        Evaluate whether a candidate satisfies all configured filter
        conditions.

        Args:
            bundle: The candidate's evidence bundle.
            composite_score: The candidate's computed composite score.
            confidence: The candidate's computed confidence level.

        Returns:
            True if the candidate passes every configured filter
            condition, False otherwise.
        """
        if (
            self.min_composite_score is not None
            and composite_score < self.min_composite_score
        ):
            return False

        if self.max_toxicity_level is not None and bundle.toxicity_prediction is not None:
            observed = self._TOXICITY_ORDER[bundle.toxicity_prediction.toxicity_level]
            allowed = self._TOXICITY_ORDER[self.max_toxicity_level]
            if observed > allowed:
                return False

        if self.require_lipinski_compliant and bundle.molecular_properties is not None:
            if bundle.molecular_properties.lipinski_violations != 0:
                return False

        if self.min_confidence is not None:
            if self._CONFIDENCE_ORDER[confidence] < self._CONFIDENCE_ORDER[self.min_confidence]:
                return False

        return all(
            predicate(bundle, composite_score, confidence)
            for predicate in self.custom_predicates
        )


# ---------------------------------------------------------------------------
# Statistics and metadata
# ---------------------------------------------------------------------------


@dataclass
class RankingStatistics:
    """
    Aggregate statistics describing a completed ranking run.

    Attributes:
        candidate_count: Number of candidates included in the statistics.
        mean_composite_score: Mean composite score across candidates.
        median_composite_score: Median composite score across candidates.
        std_dev_composite_score: Population standard deviation of
            composite scores (0.0 if fewer than two candidates).
        min_composite_score: Minimum composite score observed.
        max_composite_score: Maximum composite score observed.
        confidence_distribution: Count of candidates falling into each
            qualitative confidence bucket (keyed by lowercase label).
        criterion_averages: Mean normalized score for each ranking
            criterion, across candidates.
        pareto_optimal_candidate_ids: Identifiers of candidates on the
            Pareto frontier -- i.e., no other candidate is at least as
            good on every criterion and strictly better on at least one.
    """

    candidate_count: int
    mean_composite_score: float
    median_composite_score: float
    std_dev_composite_score: float
    min_composite_score: float
    max_composite_score: float
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    criterion_averages: dict[str, float] = field(default_factory=dict)
    pareto_optimal_candidate_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate non-negative counts and standard deviation."""
        if self.candidate_count < 0:
            raise ValueError("candidate_count cannot be negative.")
        if self.std_dev_composite_score < 0:
            raise ValueError("std_dev_composite_score cannot be negative.")


@dataclass
class RankingMetadata:
    """
    Metadata describing how a ranking run was configured and executed,
    useful for reproducibility, auditing, and report generation.

    Attributes:
        run_id: Unique identifier for the ranking run.
        strategy_name: Name of the ``RankingStrategy`` used to combine
            criteria.
        normalization_strategy: The normalization strategy applied to raw
            criterion values.
        weights: The (unnormalized) weight used for each criterion.
        candidate_pool_size: Number of candidates submitted for ranking.
        filtered_out_count: Number of candidates excluded by the
            configured ``RankingFilter``.
        ranked_candidate_count: Number of candidates present in the final
            ranking (after filtering and top-k truncation).
        top_k: The ``top_k`` truncation applied, if any.
        created_at: UTC timestamp when the ranking run was executed.
    """

    run_id: str
    strategy_name: str
    normalization_strategy: NormalizationStrategy
    weights: dict[str, float]
    candidate_pool_size: int
    filtered_out_count: int
    ranked_candidate_count: int
    top_k: int | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class ScoreNormalizer:
    """
    Normalizes raw per-criterion values, collected across a candidate
    pool, onto a common [0.0, 1.0] scale, honoring each criterion's
    ``higher_is_better`` direction.
    """

    @staticmethod
    def _min_max(values: dict[str, float]) -> dict[str, float]:
        """
        Rescale values linearly so the minimum maps to 0.0 and the
        maximum maps to 1.0.

        Args:
            values: Mapping of candidate ID to raw value.

        Returns:
            Mapping of candidate ID to rescaled value in [0.0, 1.0]. If
            all values are equal, every candidate receives 1.0 (no
            differentiating information; treated as neutral-favorable
            rather than penalizing every candidate equally toward zero).
        """
        if not values:
            return {}
        low = min(values.values())
        high = max(values.values())
        if high == low:
            return {candidate_id: 1.0 for candidate_id in values}
        span = high - low
        return {
            candidate_id: (value - low) / span for candidate_id, value in values.items()
        }

    @staticmethod
    def _clamp(values: dict[str, float]) -> dict[str, float]:
        """
        Clip values into [0.0, 1.0] without rescaling.

        Args:
            values: Mapping of candidate ID to raw value.

        Returns:
            Mapping of candidate ID to clamped value in [0.0, 1.0].
        """
        return {
            candidate_id: min(1.0, max(0.0, value))
            for candidate_id, value in values.items()
        }

    @staticmethod
    def _z_score(values: dict[str, float]) -> dict[str, float]:
        """
        Standardize values to a z-score, then squash to [0.0, 1.0] via a
        logistic function.

        Args:
            values: Mapping of candidate ID to raw value.

        Returns:
            Mapping of candidate ID to squashed value in [0.0, 1.0]. If
            fewer than two distinct values are present, every candidate
            receives 1.0.
        """
        if not values:
            return {}
        if len(set(values.values())) < 2:
            return {candidate_id: 1.0 for candidate_id in values}

        mean = statistics.mean(values.values())
        std_dev = statistics.pstdev(values.values())
        if std_dev == 0:
            return {candidate_id: 1.0 for candidate_id in values}

        return {
            candidate_id: 1.0 / (1.0 + math.exp(-((value - mean) / std_dev)))
            for candidate_id, value in values.items()
        }

    @classmethod
    def normalize(
        cls,
        values: dict[str, float],
        strategy: NormalizationStrategy,
        higher_is_better: bool,
    ) -> dict[str, float]:
        """
        Normalize a criterion's raw values across a candidate pool,
        applying direction correction so that higher normalized values
        always represent more favorable outcomes.

        Args:
            values: Mapping of candidate ID to raw criterion value.
            strategy: The normalization strategy to apply.
            higher_is_better: Whether higher raw values are favorable. If
                False, the normalized scale is inverted after
                normalization.

        Returns:
            Mapping of candidate ID to normalized value in [0.0, 1.0].

        Raises:
            RankingConfigurationError: If an unrecognized normalization
                strategy is supplied.
        """
        if strategy == NormalizationStrategy.MIN_MAX:
            normalized = cls._min_max(values)
        elif strategy == NormalizationStrategy.CLAMP:
            normalized = cls._clamp(values)
        elif strategy == NormalizationStrategy.Z_SCORE:
            normalized = cls._z_score(values)
        else:
            raise RankingConfigurationError(
                f"Unrecognized normalization strategy: {strategy}"
            )

        if not higher_is_better:
            normalized = {
                candidate_id: 1.0 - value for candidate_id, value in normalized.items()
            }

        return normalized


# ---------------------------------------------------------------------------
# Ranking strategies (multi-objective combination)
# ---------------------------------------------------------------------------


class RankingStrategy(ABC):
    """
    Abstract interface for combining normalized, per-criterion scores
    into a single composite score.

    Implementing this interface allows alternative multi-objective
    combination schemes -- or, in the future, a learned aggregation
    model -- to be used interchangeably with ``CandidateRankingEngine``
    without changing its public API.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Return a short, human-readable identifier for this strategy.

        Returns:
            The strategy's name (e.g., "weighted_sum",
            "weighted_geometric_mean").
        """
        raise NotImplementedError

    @abstractmethod
    def combine(
        self, normalized_scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """
        Combine normalized per-criterion scores into a single composite
        score.

        Args:
            normalized_scores: Mapping of criterion name to its
                normalized score, in [0.0, 1.0], for a single candidate.
            weights: Mapping of criterion name to its (unnormalized)
                weight.

        Returns:
            A composite score, in [0.0, 1.0].
        """
        raise NotImplementedError


class WeightedSumRankingStrategy(RankingStrategy):
    """
    Combines criteria via a weight-normalized linear (weighted sum)
    combination. This is the default, most interpretable strategy: each
    criterion's contribution is directly proportional to its share of
    total weight.
    """

    @property
    def strategy_name(self) -> str:
        """Return the strategy's identifier."""
        return "weighted_sum"

    def combine(
        self, normalized_scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """
        Compute a weighted arithmetic mean of the normalized scores.

        Args:
            normalized_scores: Mapping of criterion name to its
                normalized score, in [0.0, 1.0].
            weights: Mapping of criterion name to its weight.

        Returns:
            The weighted arithmetic mean, in [0.0, 1.0].
        """
        applicable = {
            name: weight for name, weight in weights.items() if name in normalized_scores
        }
        total_weight = sum(applicable.values())
        if total_weight <= 0:
            return 0.0
        weighted_total = sum(
            normalized_scores[name] * weight for name, weight in applicable.items()
        )
        return min(1.0, max(0.0, weighted_total / total_weight))


class WeightedGeometricMeanRankingStrategy(RankingStrategy):
    """
    Combines criteria via a weight-normalized geometric mean. Unlike the
    weighted sum, a very poor score on any single heavily-weighted
    criterion sharply penalizes the composite score, which is often
    desirable for multi-objective drug discovery ranking (e.g., a highly
    toxic compound should not be rescued by an excellent similarity
    score).

    A small epsilon is used in place of zero to avoid collapsing the
    entire composite score to zero from a single criterion, which would
    make ranking among "zero-scoring-on-one-axis" candidates impossible.
    """

    _EPSILON: float = 1e-6

    @property
    def strategy_name(self) -> str:
        """Return the strategy's identifier."""
        return "weighted_geometric_mean"

    def combine(
        self, normalized_scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """
        Compute a weighted geometric mean of the normalized scores.

        Args:
            normalized_scores: Mapping of criterion name to its
                normalized score, in [0.0, 1.0].
            weights: Mapping of criterion name to its weight.

        Returns:
            The weighted geometric mean, in [0.0, 1.0].
        """
        applicable = {
            name: weight for name, weight in weights.items() if name in normalized_scores
        }
        total_weight = sum(applicable.values())
        if total_weight <= 0:
            return 0.0

        log_sum = 0.0
        for name, weight in applicable.items():
            value = max(self._EPSILON, normalized_scores[name])
            log_sum += (weight / total_weight) * math.log(value)
        return min(1.0, max(0.0, math.exp(log_sum)))


# ---------------------------------------------------------------------------
# Ranking engine (orchestration layer)
# ---------------------------------------------------------------------------


class CandidateRankingEngine:
    """
    Orchestrates evidence extraction, normalization, multi-objective
    scoring, filtering, and reporting to rank candidate compounds.

    This is the primary integration point for ``pipeline.py`` once
    per-candidate evidence has been collected from
    ``molecular_properties``, ``compound_similarity``,
    ``target_prediction``, ``toxicity_prediction``, and
    ``drug_repurposing``.

    Attributes:
        config: The active ``CandidateRankingConfig``.
        strategy: The active ``RankingStrategy`` used to combine
            normalized criteria into a composite score.
    """

    def __init__(
        self,
        config: CandidateRankingConfig | None = None,
        strategy: RankingStrategy | None = None,
    ) -> None:
        """
        Initialize the candidate ranking engine.

        Args:
            config: The ranking configuration to use. If omitted, the
                default configuration (equal weighting across all
                built-in criteria, min-max normalization) is used.
            strategy: The ranking combination strategy to use. If
                omitted, ``WeightedSumRankingStrategy`` is used.
        """
        self.config = config or CandidateRankingConfig.default()
        self.strategy = strategy or WeightedSumRankingStrategy()

    @staticmethod
    def _extract_raw_scores(
        bundle: CandidateEvidenceBundle,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """
        Extract raw criterion values and explanatory notes from a
        candidate's evidence bundle.

        Criteria for which no supporting evidence is available are
        simply omitted from the returned mapping (rather than defaulted
        to an arbitrary value), so that normalization and weighting are
        only ever performed over criteria with real data.

        Args:
            bundle: The candidate's evidence bundle.

        Returns:
            A tuple of (raw scores by criterion name, explanatory note by
            criterion name).
        """
        raw_scores: dict[str, float] = {}
        notes: dict[str, str] = {}

        if bundle.similar_compound is not None:
            raw_scores["structural_similarity"] = bundle.similar_compound.similarity_score
            notes["structural_similarity"] = (
                f"Structural similarity to reference compound: "
                f"{bundle.similar_compound.similarity_score:.2f} "
                f"({bundle.similar_compound.metric.value})."
            )

        if bundle.target_predictions:
            best_target = max(
                bundle.target_predictions, key=lambda tp: tp.confidence_score
            )
            raw_scores["target_confidence"] = best_target.confidence_score
            notes["target_confidence"] = (
                f"Strongest predicted target interaction: "
                f"{best_target.target.name} "
                f"(confidence {best_target.confidence_score:.2f})."
            )

        if bundle.toxicity_prediction is not None:
            raw_scores["toxicity_risk"] = bundle.toxicity_prediction.toxicity_score
            notes["toxicity_risk"] = (
                f"Predicted toxicity risk: "
                f"{bundle.toxicity_prediction.toxicity_score:.2f} "
                f"({bundle.toxicity_prediction.toxicity_level.value})."
            )

        if bundle.molecular_properties is not None:
            if bundle.molecular_properties.qed_score is not None:
                raw_scores["drug_likeness"] = bundle.molecular_properties.qed_score
                notes["drug_likeness"] = (
                    f"QED drug-likeness score: "
                    f"{bundle.molecular_properties.qed_score:.2f}."
                )
            raw_scores["lipinski_violations"] = float(
                bundle.molecular_properties.lipinski_violations
            )
            notes["lipinski_violations"] = (
                f"Lipinski Rule of Five violations: "
                f"{bundle.molecular_properties.lipinski_violations}."
            )

        if bundle.repurposing_score is not None:
            raw_scores["repurposing_evidence"] = bundle.repurposing_score
            confidence_note = (
                f" (confidence: {bundle.repurposing_confidence.value})"
                if bundle.repurposing_confidence is not None
                else ""
            )
            notes["repurposing_evidence"] = (
                f"Composite drug repurposing evidence score: "
                f"{bundle.repurposing_score:.2f}{confidence_note}."
            )

        for name, value in bundle.custom_scores.items():
            raw_scores[name] = value
            notes[name] = f"Custom criterion '{name}': {value:.2f}."

        return raw_scores, notes

    @staticmethod
    def _confidence_for_candidate(
        bundle: CandidateEvidenceBundle, criteria_covered: int, criteria_total: int
    ) -> PredictionConfidence:
        """
        Estimate the overall confidence of a candidate's ranking, based
        on how much of the configured evidence was actually available
        and on the confidence levels of any underlying predictions.

        Args:
            bundle: The candidate's evidence bundle.
            criteria_covered: Number of configured criteria for which
                evidence was actually available.
            criteria_total: Total number of configured criteria.

        Returns:
            A ``PredictionConfidence`` bucket reflecting overall
            evidentiary support for the ranking.
        """
        coverage_ratio = criteria_covered / criteria_total if criteria_total else 0.0

        confidence_rank = {
            PredictionConfidence.LOW: 0,
            PredictionConfidence.MEDIUM: 1,
            PredictionConfidence.HIGH: 2,
            PredictionConfidence.VERY_HIGH: 3,
        }
        confidence_values: list[int] = []
        if bundle.target_predictions:
            best_target = max(
                bundle.target_predictions, key=lambda tp: tp.confidence_score
            )
            confidence_values.append(confidence_rank[best_target.confidence_level])
        if bundle.toxicity_prediction is not None:
            confidence_values.append(
                confidence_rank[bundle.toxicity_prediction.confidence]
            )
        if bundle.repurposing_confidence is not None:
            confidence_values.append(confidence_rank[bundle.repurposing_confidence])

        underlying_average = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 1.0
        )

        combined_score = 0.5 * coverage_ratio + 0.5 * (underlying_average / 3.0)

        if combined_score >= 0.85:
            return PredictionConfidence.VERY_HIGH
        if combined_score >= 0.65:
            return PredictionConfidence.HIGH
        if combined_score >= 0.4:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def rank(
        self,
        bundles: list[CandidateEvidenceBundle],
        top_k: int | None = None,
        ranking_filter: RankingFilter | None = None,
    ) -> DrugDiscoveryResult:
        """
        Score, filter, and rank a pool of candidates.

        Args:
            bundles: The candidates' evidence bundles.
            top_k: If provided, limits the number of ranked candidates
                returned to the top ``top_k`` highest composite scores.
            ranking_filter: Optional filtering criteria applied after
                scoring, before top-k truncation.

        Returns:
            A ``DrugDiscoveryResult`` containing one ``DrugCandidate`` and
            one ``CandidateRanking`` per retained candidate, sorted from
            best to worst, with ranking statistics and metadata attached
            under ``metadata["statistics"]`` and
            ``metadata["ranking_metadata"]``.

        Raises:
            RankingConfigurationError: If no criteria are configured or
                total weight is non-positive.
            CandidateRankingError: If scoring fails unexpectedly.
            ValueError: If ``top_k`` is out of valid range.
        """
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be a positive integer when provided.")

        run_id = str(uuid.uuid4())
        logger.info(
            "Starting candidate ranking run_id='%s' for %d candidate(s) "
            "using strategy '%s'.",
            run_id,
            len(bundles),
            self.strategy.strategy_name,
        )

        try:
            raw_scores_by_candidate: dict[str, dict[str, float]] = {}
            notes_by_candidate: dict[str, dict[str, str]] = {}
            for bundle in bundles:
                raw_scores, notes = self._extract_raw_scores(bundle)
                raw_scores_by_candidate[bundle.compound.compound_id] = raw_scores
                notes_by_candidate[bundle.compound.compound_id] = notes

            criterion_names = [criterion.name for criterion in self.config.criteria]
            normalized_by_candidate: dict[str, dict[str, float]] = {
                candidate_id: {} for candidate_id in raw_scores_by_candidate
            }
            for criterion in self.config.criteria:
                values_for_criterion = {
                    candidate_id: raw_scores[criterion.name]
                    for candidate_id, raw_scores in raw_scores_by_candidate.items()
                    if criterion.name in raw_scores
                }
                if not values_for_criterion:
                    continue
                normalized_values = ScoreNormalizer.normalize(
                    values_for_criterion,
                    self.config.normalization_strategy,
                    criterion.higher_is_better,
                )
                for candidate_id, normalized_value in normalized_values.items():
                    normalized_by_candidate[candidate_id][criterion.name] = normalized_value

            weights = self.config.weights
        except RankingConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception("Candidate ranking evidence processing failed.")
            raise CandidateRankingError(
                "Failed to extract or normalize candidate evidence."
            ) from exc

        bundles_by_id = {bundle.compound.compound_id: bundle for bundle in bundles}
        scored_entries: list[
            tuple[CandidateEvidenceBundle, float, PredictionConfidence, dict[str, float]]
        ] = []

        for candidate_id, normalized_scores in normalized_by_candidate.items():
            bundle = bundles_by_id[candidate_id]
            try:
                composite_score = self.strategy.combine(normalized_scores, weights)
            except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
                logger.exception(
                    "Composite score calculation failed for candidate '%s'.",
                    candidate_id,
                )
                raise CandidateRankingError(
                    f"Composite score calculation failed for candidate "
                    f"'{candidate_id}'."
                ) from exc

            confidence = self._confidence_for_candidate(
                bundle,
                criteria_covered=len(normalized_scores),
                criteria_total=len(criterion_names),
            )
            scored_entries.append((bundle, composite_score, confidence, normalized_scores))

        filtered_entries = [
            entry
            for entry in scored_entries
            if ranking_filter is None
            or ranking_filter.is_satisfied_by(entry[0], entry[1], entry[2])
        ]
        filtered_out_count = len(scored_entries) - len(filtered_entries)

        # Deterministic ordering: composite score descending, then
        # candidate_id ascending as a stable tie-breaker.
        filtered_entries.sort(
            key=lambda entry: (-entry[1], entry[0].compound.compound_id)
        )

        if top_k is not None:
            filtered_entries = filtered_entries[:top_k]

        drug_candidates: list[DrugCandidate] = []
        rankings: list[CandidateRanking] = []
        for rank_position, (bundle, composite_score, confidence, normalized_scores) in enumerate(
            filtered_entries, start=1
        ):
            candidate_id = bundle.compound.compound_id
            explanations = self._build_explanations(
                notes_by_candidate.get(candidate_id, {}),
                normalized_scores,
                weights,
                composite_score,
                confidence,
            )

            drug_candidates.append(
                DrugCandidate(
                    candidate_id=f"RANKED_{candidate_id}",
                    compound=bundle.compound,
                    molecular_properties=bundle.molecular_properties,
                    target_predictions=bundle.target_predictions,
                    toxicity_prediction=bundle.toxicity_prediction,
                    status=CandidateStatus.PENDING,
                    overall_score=composite_score,
                    notes=" ".join(explanations),
                )
            )
            rankings.append(
                CandidateRanking(
                    candidate_id=f"RANKED_{candidate_id}",
                    rank=rank_position,
                    composite_score=composite_score,
                    criteria_scores=normalized_scores,
                    notes=f"Confidence: {confidence.value}. " + " ".join(explanations),
                )
            )

        ranking_statistics = self.compute_statistics(rankings)
        ranking_metadata = RankingMetadata(
            run_id=run_id,
            strategy_name=self.strategy.strategy_name,
            normalization_strategy=self.config.normalization_strategy,
            weights=weights,
            candidate_pool_size=len(bundles),
            filtered_out_count=filtered_out_count,
            ranked_candidate_count=len(drug_candidates),
            top_k=top_k,
        )

        completion_timestamp = datetime.now(timezone.utc)
        result = DrugDiscoveryResult(
            run_id=run_id,
            candidates=drug_candidates,
            rankings=rankings,
            stage=DiscoveryStage.CANDIDATE_RANKING,
            created_at=completion_timestamp,
            completed_at=completion_timestamp,
            metadata={
                "statistics": ranking_statistics,
                "ranking_metadata": ranking_metadata,
            },
        )

        logger.info(
            "Completed candidate ranking run_id='%s': %d candidate(s) "
            "ranked, %d filtered out.",
            run_id,
            len(drug_candidates),
            filtered_out_count,
        )
        return result

    @staticmethod
    def _build_explanations(
        notes: dict[str, str],
        normalized_scores: dict[str, float],
        weights: dict[str, float],
        composite_score: float,
        confidence: PredictionConfidence,
    ) -> list[str]:
        """
        Build a list of human-readable explanations describing how a
        candidate's composite score was derived.

        Args:
            notes: Per-criterion explanatory notes from raw score
                extraction.
            normalized_scores: Per-criterion normalized scores actually
                used in scoring.
            weights: Per-criterion weights.
            composite_score: The candidate's final composite score.
            confidence: The candidate's computed confidence level.

        Returns:
            A list of explanation strings, ordered from overall summary
            to specific per-criterion contributions.
        """
        explanations = [
            f"Composite ranking score is {composite_score:.2f} "
            f"(confidence: {confidence.value})."
        ]
        total_weight = sum(
            weights[name] for name in normalized_scores if name in weights
        )
        for name, normalized_value in sorted(
            normalized_scores.items(),
            key=lambda item: weights.get(item[0], 0.0),
            reverse=True,
        ):
            weight = weights.get(name, 0.0)
            share = (weight / total_weight * 100) if total_weight > 0 else 0.0
            base_note = notes.get(name, f"Criterion '{name}'.")
            explanations.append(
                f"{base_note} Normalized score {normalized_value:.2f}, "
                f"contributing ~{share:.0f}% of total weight."
            )
        return explanations

    @staticmethod
    def compute_statistics(rankings: list[CandidateRanking]) -> RankingStatistics:
        """
        Compute aggregate statistics over a completed set of rankings.

        Args:
            rankings: The final ``CandidateRanking`` list, each carrying
                its per-criterion normalized scores in
                ``criteria_scores`` and its confidence embedded in
                ``notes`` (as produced by ``rank``).

        Returns:
            A populated ``RankingStatistics`` instance.
        """
        if not rankings:
            return RankingStatistics(
                candidate_count=0,
                mean_composite_score=0.0,
                median_composite_score=0.0,
                std_dev_composite_score=0.0,
                min_composite_score=0.0,
                max_composite_score=0.0,
            )

        scores = [ranking.composite_score for ranking in rankings]

        confidence_distribution: dict[str, int] = {}
        for ranking in rankings:
            level = "unknown"
            if ranking.notes and "Confidence:" in ranking.notes:
                fragment = ranking.notes.split("Confidence:", 1)[1].strip()
                level = fragment.split(".", 1)[0].strip().lower() or "unknown"
            confidence_distribution[level] = confidence_distribution.get(level, 0) + 1

        criterion_totals: dict[str, list[float]] = {}
        for ranking in rankings:
            for name, value in ranking.criteria_scores.items():
                criterion_totals.setdefault(name, []).append(value)
        criterion_averages = {
            name: sum(values) / len(values) for name, values in criterion_totals.items()
        }

        pareto_ids = CandidateRankingEngine._compute_pareto_frontier(rankings)

        return RankingStatistics(
            candidate_count=len(rankings),
            mean_composite_score=statistics.mean(scores),
            median_composite_score=statistics.median(scores),
            std_dev_composite_score=(
                statistics.pstdev(scores) if len(scores) > 1 else 0.0
            ),
            min_composite_score=min(scores),
            max_composite_score=max(scores),
            confidence_distribution=confidence_distribution,
            criterion_averages=criterion_averages,
            pareto_optimal_candidate_ids=pareto_ids,
        )

    @staticmethod
    def _compute_pareto_frontier(rankings: list[CandidateRanking]) -> list[str]:
        """
        Identify the Pareto-optimal (non-dominated) subset of candidates
        based on their per-criterion normalized scores.

        A candidate A is dominated by candidate B if B is at least as
        good as A on every criterion and strictly better on at least one.
        Non-dominated candidates form the Pareto frontier and represent
        the set of "best trade-off" compounds when no single weighting
        scheme is assumed to be authoritative.

        Args:
            rankings: The final ``CandidateRanking`` list, each carrying
                its per-criterion normalized scores.

        Returns:
            A list of candidate IDs on the Pareto frontier, in no
            particular order.
        """
        pareto_ids: list[str] = []
        for candidate in rankings:
            dominated = False
            for other in rankings:
                if other.candidate_id == candidate.candidate_id:
                    continue
                if CandidateRankingEngine._dominates(
                    other.criteria_scores, candidate.criteria_scores
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_ids.append(candidate.candidate_id)
        return pareto_ids

    @staticmethod
    def _dominates(scores_a: dict[str, float], scores_b: dict[str, float]) -> bool:
        """
        Determine whether criteria scores A dominate criteria scores B in
        the Pareto sense (at least as good on every shared criterion, and
        strictly better on at least one).

        Args:
            scores_a: Per-criterion normalized scores for candidate A.
            scores_b: Per-criterion normalized scores for candidate B.

        Returns:
            True if A dominates B, False otherwise.
        """
        shared_criteria = set(scores_a) & set(scores_b)
        if not shared_criteria:
            return False
        at_least_as_good = all(
            scores_a[name] >= scores_b[name] for name in shared_criteria
        )
        strictly_better = any(
            scores_a[name] > scores_b[name] for name in shared_criteria
        )
        return at_least_as_good and strictly_better

    @staticmethod
    def generate_report(result: DrugDiscoveryResult, top_n: int = 10) -> str:
        """
        Generate a human-readable Markdown report summarizing a ranking
        run's methodology, statistics, and top candidates.

        Args:
            result: A ``DrugDiscoveryResult`` produced by ``rank``.
            top_n: Number of top-ranked candidates to include in detail.

        Returns:
            A Markdown-formatted report string.
        """
        stats: RankingStatistics | None = result.metadata.get("statistics")
        run_metadata: RankingMetadata | None = result.metadata.get("ranking_metadata")

        lines: list[str] = ["# Drug Candidate Ranking Report", ""]
        lines.append(f"**Run ID:** `{result.run_id}`")
        if run_metadata is not None:
            lines.append(f"**Strategy:** {run_metadata.strategy_name}")
            lines.append(
                f"**Normalization:** {run_metadata.normalization_strategy.value}"
            )
            lines.append(
                f"**Candidates evaluated:** {run_metadata.candidate_pool_size} "
                f"(filtered out: {run_metadata.filtered_out_count}, "
                f"ranked: {run_metadata.ranked_candidate_count})"
            )
            weights_summary = ", ".join(
                f"{name}={weight:.2f}" for name, weight in run_metadata.weights.items()
            )
            lines.append(f"**Criterion weights:** {weights_summary}")
        lines.append("")

        if stats is not None and stats.candidate_count > 0:
            lines.append("## Summary Statistics")
            lines.append(f"- Candidates ranked: {stats.candidate_count}")
            lines.append(f"- Mean composite score: {stats.mean_composite_score:.3f}")
            lines.append(
                f"- Median composite score: {stats.median_composite_score:.3f}"
            )
            lines.append(f"- Std. deviation: {stats.std_dev_composite_score:.3f}")
            lines.append(
                f"- Range: {stats.min_composite_score:.3f} - "
                f"{stats.max_composite_score:.3f}"
            )
            if stats.confidence_distribution:
                distribution_summary = ", ".join(
                    f"{level}: {count}"
                    for level, count in stats.confidence_distribution.items()
                )
                lines.append(f"- Confidence distribution: {distribution_summary}")
            if stats.pareto_optimal_candidate_ids:
                lines.append(
                    f"- Pareto-optimal candidates: "
                    f"{', '.join(stats.pareto_optimal_candidate_ids)}"
                )
            lines.append("")

        lines.append(f"## Top {min(top_n, len(result.rankings))} Candidates")
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in result.candidates
        }
        for ranking in result.rankings[:top_n]:
            candidate = candidates_by_id.get(ranking.candidate_id)
            compound_name = candidate.compound.name if candidate else ranking.candidate_id
            lines.append(
                f"### {ranking.rank}. {compound_name} "
                f"(score: {ranking.composite_score:.3f})"
            )
            if ranking.notes:
                lines.append(ranking.notes)
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def rank_candidates(
    bundles: list[CandidateEvidenceBundle],
    config: CandidateRankingConfig | None = None,
    strategy: RankingStrategy | None = None,
    top_k: int | None = None,
    ranking_filter: RankingFilter | None = None,
) -> DrugDiscoveryResult:
    """
    Score, filter, and rank a pool of candidates using a default
    ``CandidateRankingEngine``.

    This is the primary entry point intended for use by ``pipeline.py``.

    Args:
        bundles: The candidates' evidence bundles.
        config: Optional ranking configuration. If omitted, the default
            configuration is used.
        strategy: Optional ranking combination strategy. If omitted,
            ``WeightedSumRankingStrategy`` is used.
        top_k: If provided, limits the number of ranked candidates
            returned to the top ``top_k`` highest composite scores.
        ranking_filter: Optional filtering criteria applied after
            scoring, before top-k truncation.

    Returns:
        A ``DrugDiscoveryResult`` containing ranked candidates, from best
        to worst, with statistics and metadata attached.

    Raises:
        RankingConfigurationError: If no criteria are configured or total
            weight is non-positive.
        CandidateRankingError: If scoring fails unexpectedly.
    """
    engine = CandidateRankingEngine(config=config, strategy=strategy)
    return engine.rank(bundles=bundles, top_k=top_k, ranking_filter=ranking_filter)