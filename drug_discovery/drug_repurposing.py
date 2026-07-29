"""
drug_discovery/drug_repurposing.py

Drug repurposing candidate scoring and ranking for the BioNexus Drug
Discovery module.

This module identifies and ranks existing compounds that may be
repurposed against a disease, target, or reference compound of
interest, by combining evidence from multiple sources:

    1. Structural similarity (reusing
       ``drug_discovery.compound_similarity``).
    2. Predicted biological target overlap (reusing
       ``drug_discovery.target_prediction``).
    3. Semantic embedding similarity (via a pluggable
       ``EmbeddingSimilarityProvider`` that adapts BioNexus's Embeddings
       module).
    4. Knowledge Graph relationships spanning diseases, genes, proteins,
       and pathways (via a pluggable ``KnowledgeGraphProvider`` that
       adapts BioNexus's Knowledge Graph module).

The Embeddings and Knowledge Graph integrations are expressed as
``typing.Protocol`` interfaces rather than direct imports of those
modules' concrete classes. This keeps ``drug_repurposing.py`` decoupled
from the internal implementation details of those (separately owned)
modules -- any object exposing the expected methods, including a thin
adapter around the real Embeddings/Knowledge Graph modules, can be
supplied at runtime.

The scoring strategy itself is pluggable via the
``RepurposingScoringEngine`` abstract interface, following the same
pattern used by ``target_prediction.py`` and ``toxicity_prediction.py``.
This allows a future, more sophisticated AI-driven scoring model to
replace ``RuleBasedRepurposingScorer`` without changing the public API
consumed by ``pipeline.py``.

Compatibility
-------------
Targets Python 3.11. Depends on ``drug_discovery.models``,
``drug_discovery.compound_similarity``, and
``drug_discovery.target_prediction``.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

from drug_discovery.compound_similarity import (
    CompoundSimilaritySearchEngine,
    FingerprintConfig,
)
from drug_discovery.models import (
    CandidateRanking,
    CandidateStatus,
    Compound,
    DiscoveryStage,
    DrugCandidate,
    DrugDiscoveryResult,
    DrugTarget,
    PredictionConfidence,
    TargetPrediction,
)
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
)
from drug_discovery.target_prediction import TargetPredictionService

__all__ = [
    "InvalidSMILESError",
    "DescriptorCalculationError",
    "RepurposingQueryError",
    "RepurposingScoringError",
    "KnowledgeGraphEntityType",
    "KnowledgeGraphRelationship",
    "EmbeddingSimilarityProvider",
    "KnowledgeGraphProvider",
    "RepurposingQuery",
    "RepurposingEvidence",
    "RepurposingScoringEngine",
    "RuleBasedRepurposingScorer",
    "DrugRepurposingService",
    "find_repurposing_candidates",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class RepurposingQueryError(ValueError):
    """
    Raised when a repurposing query is malformed or under-specified
    (e.g., no disease name, compound, SMILES, or target was provided).
    """


class RepurposingScoringError(RuntimeError):
    """
    Raised when candidate scoring fails unexpectedly for an otherwise
    valid query and candidate.

    Attributes:
        candidate_id: Identifier of the candidate compound being scored
            when the failure occurred.
        engine_name: Name of the scoring engine that raised the failure.
    """

    def __init__(
        self,
        candidate_id: str,
        engine_name: str,
        message: str | None = None,
    ) -> None:
        self.candidate_id = candidate_id
        self.engine_name = engine_name
        resolved_message = (
            message
            or f"Repurposing score calculation failed for "
            f"candidate_id='{candidate_id}' using engine '{engine_name}'."
        )
        super().__init__(resolved_message)


# ---------------------------------------------------------------------------
# Knowledge Graph integration types
# ---------------------------------------------------------------------------


class KnowledgeGraphEntityType(str, Enum):
    """Entity types recognized when querying the BioNexus Knowledge Graph."""

    DISEASE = "disease"
    GENE = "gene"
    PROTEIN = "protein"
    PATHWAY = "pathway"
    DRUG = "drug"
    TARGET = "target"


@dataclass(frozen=True)
class KnowledgeGraphRelationship:
    """
    A single relationship (edge or resolved path) between two entities in
    the BioNexus Knowledge Graph, relevant to a repurposing decision.

    Attributes:
        source_id: Identifier of the source entity (typically the
            candidate drug).
        source_type: Entity type of the source.
        target_id: Identifier of the target entity (e.g., a disease,
            gene, protein, or pathway).
        target_type: Entity type of the target.
        relation_label: Human-readable label describing the relationship
            (e.g., "treats", "upregulates", "participates_in").
        strength: Normalized relationship strength/confidence, in
            [0.0, 1.0], as reported by the Knowledge Graph.
        path_description: Human-readable description of the relationship
            or the multi-hop path it represents (e.g., "Drug -> inhibits
            -> Protein -> encoded_by -> Gene -> associated_with ->
            Disease").
    """

    source_id: str
    source_type: KnowledgeGraphEntityType
    target_id: str
    target_type: KnowledgeGraphEntityType
    relation_label: str
    strength: float
    path_description: str

    def __post_init__(self) -> None:
        """Validate the relationship strength range."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be within [0.0, 1.0].")


@runtime_checkable
class EmbeddingSimilarityProvider(Protocol):
    """
    Structural interface for an adapter over BioNexus's Embeddings
    module, used to compute semantic similarity between two named
    entities (e.g., a disease and a compound, or two compounds).

    Any object exposing a compatible ``get_similarity`` method -- for
    example, a thin wrapper around the real Embeddings module's vector
    store and similarity search -- may be supplied wherever this
    protocol is required.
    """

    def get_similarity(
        self,
        entity_a_id: str,
        entity_a_type: str,
        entity_b_id: str,
        entity_b_type: str,
    ) -> float:
        """
        Compute the semantic embedding similarity between two entities.

        Args:
            entity_a_id: Identifier or name of the first entity (e.g., a
                disease name or compound ID).
            entity_a_type: Type label of the first entity (e.g.,
                "disease", "drug").
            entity_b_id: Identifier or name of the second entity.
            entity_b_type: Type label of the second entity.

        Returns:
            A similarity score in [0.0, 1.0].
        """
        ...


@runtime_checkable
class KnowledgeGraphProvider(Protocol):
    """
    Structural interface for an adapter over BioNexus's Knowledge Graph
    module, used to retrieve relationships connecting a candidate drug
    to a disease, gene, protein, pathway, or target of interest.

    Any object exposing a compatible ``get_relationships`` method -- for
    example, a thin wrapper around the real Knowledge Graph module's
    query engine -- may be supplied wherever this protocol is required.
    """

    def get_relationships(
        self,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
    ) -> list[KnowledgeGraphRelationship]:
        """
        Retrieve relationships (direct edges or resolved multi-hop
        paths) connecting a source entity to a target entity.

        Args:
            source_id: Identifier of the source entity (typically the
                candidate drug's compound ID).
            source_type: Type label of the source entity.
            target_id: Identifier or name of the target entity (e.g., a
                disease name, gene symbol, or target ID).
            target_type: Type label of the target entity.

        Returns:
            A list of ``KnowledgeGraphRelationship`` instances describing
            how the two entities are connected. An empty list indicates
            no known relationship was found.
        """
        ...


# ---------------------------------------------------------------------------
# Query and evidence models
# ---------------------------------------------------------------------------


@dataclass
class RepurposingQuery:
    """
    A normalized repurposing query, accepting any combination of a
    disease name, reference compound, reference SMILES string, or
    biological target of interest.

    At least one of ``disease_name``, ``query_compound``,
    ``query_smiles``, or ``query_target`` must be provided.

    Attributes:
        disease_name: Name of the disease being targeted for
            repurposing (e.g., "type 2 diabetes mellitus").
        query_compound: A reference ``Compound`` whose profile candidates
            should be compared against.
        query_smiles: A reference SMILES string, used when only a raw
            structure (not a full ``Compound`` record) is available.
            Ignored if ``query_compound`` is provided.
        query_target: A specific ``DrugTarget`` of interest, used to
            focus target-overlap scoring on that target rather than on
            the full predicted target profile of ``query_compound``.
    """

    disease_name: str | None = None
    query_compound: Compound | None = None
    query_smiles: str | None = None
    query_target: DrugTarget | None = None

    def __post_init__(self) -> None:
        """Validate that at least one query input was supplied."""
        if not any(
            [
                self.disease_name,
                self.query_compound,
                self.query_smiles,
                self.query_target,
            ]
        ):
            raise RepurposingQueryError(
                "At least one of disease_name, query_compound, "
                "query_smiles, or query_target must be provided."
            )

    @property
    def effective_smiles(self) -> str | None:
        """
        Return the SMILES string to use for structural/target-based
        scoring, preferring ``query_compound.smiles`` over
        ``query_smiles`` when both are present.

        Returns:
            The effective SMILES string, or None if neither a query
            compound nor a raw query SMILES string was provided.
        """
        if self.query_compound is not None:
            return self.query_compound.smiles
        return self.query_smiles

    @property
    def effective_compound_id(self) -> str | None:
        """
        Return an identifier for the query compound suitable for
        embedding/knowledge-graph lookups.

        Returns:
            ``query_compound.compound_id`` if a query compound was
            provided, otherwise None.
        """
        if self.query_compound is not None:
            return self.query_compound.compound_id
        return None


@dataclass
class RepurposingEvidence:
    """
    A detailed, multi-source evidence breakdown supporting a single
    candidate drug's repurposing score.

    Attributes:
        candidate_id: Identifier of the scored candidate compound.
        structural_similarity_score: Structural (Tanimoto) similarity
            score relative to the query, in [0.0, 1.0].
        target_overlap_score: Score reflecting overlap between the
            candidate's predicted targets and the query's predicted or
            specified target(s), in [0.0, 1.0].
        embedding_similarity_score: Semantic embedding similarity score,
            in [0.0, 1.0].
        knowledge_graph_score: Aggregated Knowledge Graph relationship
            strength score, in [0.0, 1.0].
        composite_score: Final weighted composite score, in [0.0, 1.0].
        confidence: Qualitative confidence bucket for the composite
            score.
        shared_targets: Target predictions shared between the query and
            the candidate, contributing to ``target_overlap_score``.
        knowledge_graph_relationships: Knowledge Graph relationships
            contributing to ``knowledge_graph_score``.
        explanations: Human-readable explanations describing how each
            evidence source contributed to the composite score.
    """

    candidate_id: str
    structural_similarity_score: float
    target_overlap_score: float
    embedding_similarity_score: float
    knowledge_graph_score: float
    composite_score: float
    confidence: PredictionConfidence
    shared_targets: list[TargetPrediction] = field(default_factory=list)
    knowledge_graph_relationships: list[KnowledgeGraphRelationship] = field(
        default_factory=list
    )
    explanations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that all component and composite scores are in range."""
        for field_name, value in (
            ("structural_similarity_score", self.structural_similarity_score),
            ("target_overlap_score", self.target_overlap_score),
            ("embedding_similarity_score", self.embedding_similarity_score),
            ("knowledge_graph_score", self.knowledge_graph_score),
            ("composite_score", self.composite_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0.0, 1.0].")


# ---------------------------------------------------------------------------
# Scoring engine interface
# ---------------------------------------------------------------------------


class RepurposingScoringEngine(ABC):
    """
    Abstract interface for drug repurposing candidate scoring backends.

    Any scoring backend -- rule-based, classical machine learning, or a
    graph/embedding-driven deep learning model -- must implement this
    interface so that it can be used interchangeably by
    ``DrugRepurposingService`` and, by extension, ``pipeline.py``.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """
        Return a short, human-readable identifier for this engine.

        Returns:
            The engine's name (e.g., "rule_based_v1",
            "gnn_repurposing_v1").
        """
        raise NotImplementedError

    @abstractmethod
    def score_candidate(
        self,
        query: RepurposingQuery,
        candidate: Compound,
        candidate_target_predictions: list[TargetPrediction],
        query_target_predictions: list[TargetPrediction],
    ) -> RepurposingEvidence:
        """
        Score a single candidate compound's suitability for repurposing
        against the given query.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate: The candidate ``Compound`` being scored.
            candidate_target_predictions: Predicted targets for the
                candidate compound.
            query_target_predictions: Predicted targets for the query
                compound (empty if the query did not include a structural
                reference).

        Returns:
            A populated ``RepurposingEvidence`` instance.

        Raises:
            RepurposingScoringError: If scoring fails unexpectedly.
        """
        raise NotImplementedError


class RuleBasedRepurposingScorer(RepurposingScoringEngine):
    """
    Baseline rule-based drug repurposing scoring engine.

    Combines four evidence sources into a composite score:

        1. Structural similarity between the candidate and the query
           compound (Tanimoto similarity over Morgan fingerprints).
        2. Overlap between the candidate's and query's predicted
           biological targets (or, when a specific target of interest is
           supplied, the candidate's predicted confidence against that
           target).
        3. Semantic embedding similarity between the query (disease or
           compound) and the candidate compound, via an injected
           ``EmbeddingSimilarityProvider``.
        4. Knowledge Graph relationship strength connecting the candidate
           to the disease of interest (or other query entity), via an
           injected ``KnowledgeGraphProvider``.

    Evidence sources that cannot be computed (e.g., no embedding
    provider was supplied, or no disease name was given) contribute a
    score of 0.0 with an explanatory note, rather than causing a
    failure -- this keeps the module usable even before all upstream
    integrations (Embeddings, Knowledge Graph) are wired in.

    Attributes:
        weights: Mapping of the four evidence source names to their
            contribution weight in the composite score. Weights sum to
            1.0.
    """

    _ENGINE_NAME: str = "rule_based_v1"

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "structural_similarity_score": 0.25,
        "target_overlap_score": 0.30,
        "embedding_similarity_score": 0.20,
        "knowledge_graph_score": 0.25,
    }

    def __init__(
        self,
        similarity_engine: CompoundSimilaritySearchEngine | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the rule-based repurposing scorer.

        Args:
            similarity_engine: Optional ``CompoundSimilaritySearchEngine``
                used for structural similarity scoring. If omitted, a
                default engine (ECFP4, 2048 bits) is constructed.
            weights: Optional custom weighting for the four evidence
                sources ("structural_similarity_score",
                "target_overlap_score", "embedding_similarity_score",
                "knowledge_graph_score"). If omitted, the default
                weighting is used. Weights should sum to approximately
                1.0.

        Raises:
            ValueError: If any provided weight is negative.
        """
        self.similarity_engine = similarity_engine or CompoundSimilaritySearchEngine(
            config=FingerprintConfig()
        )
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)
        for source, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Weight for '{source}' cannot be negative.")

    @property
    def engine_name(self) -> str:
        """Return the engine's identifier."""
        return self._ENGINE_NAME

    @staticmethod
    def _score_to_confidence(composite_score: float) -> PredictionConfidence:
        """
        Map a composite repurposing score to a qualitative confidence
        bucket.

        Args:
            composite_score: Composite repurposing score, in [0.0, 1.0].

        Returns:
            The corresponding ``PredictionConfidence`` bucket.
        """
        if composite_score >= 0.85:
            return PredictionConfidence.VERY_HIGH
        if composite_score >= 0.65:
            return PredictionConfidence.HIGH
        if composite_score >= 0.4:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def _score_structural_similarity(
        self, query: RepurposingQuery, candidate: Compound, explanations: list[str]
    ) -> float:
        """
        Compute the structural similarity contribution for a candidate.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate: The candidate ``Compound`` being scored.
            explanations: List to which a human-readable explanation of
                this score is appended, in place.

        Returns:
            The structural similarity score, in [0.0, 1.0].
        """
        query_smiles = query.effective_smiles
        if not query_smiles:
            explanations.append(
                "Structural similarity not evaluated: no reference "
                "compound or SMILES was provided in the query."
            )
            return 0.0

        try:
            query_fingerprint = self.similarity_engine.generate_fingerprint(
                query_smiles
            )
            candidate_fingerprint = self.similarity_engine.generate_fingerprint(
                candidate.smiles
            )
        except InvalidSMILESError:
            explanations.append(
                f"Structural similarity not evaluated: candidate "
                f"'{candidate.compound_id}' has an invalid SMILES string."
            )
            return 0.0

        score = self.similarity_engine.compute_similarity(
            query_fingerprint, candidate_fingerprint
        )
        explanations.append(
            f"Structural (Tanimoto) similarity to the query compound is "
            f"{score:.2f}."
        )
        return score

    @staticmethod
    def _score_target_overlap(
        query: RepurposingQuery,
        candidate_target_predictions: list[TargetPrediction],
        query_target_predictions: list[TargetPrediction],
        explanations: list[str],
    ) -> tuple[float, list[TargetPrediction]]:
        """
        Compute the predicted-target overlap contribution for a
        candidate.

        If ``query.query_target`` is set, the score reflects the
        candidate's predicted confidence against that specific target.
        Otherwise, the score reflects confidence-weighted overlap
        between the candidate's and query's full predicted target sets.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate_target_predictions: Predicted targets for the
                candidate compound.
            query_target_predictions: Predicted targets for the query
                compound.
            explanations: List to which a human-readable explanation of
                this score is appended, in place.

        Returns:
            A tuple of the target overlap score, in [0.0, 1.0], and the
            list of shared/matching ``TargetPrediction`` instances.
        """
        if query.query_target is not None:
            matches = [
                prediction
                for prediction in candidate_target_predictions
                if prediction.target.target_id == query.query_target.target_id
            ]
            if not matches:
                explanations.append(
                    f"No predicted interaction was found between this "
                    f"candidate and the specified target of interest "
                    f"('{query.query_target.name}')."
                )
                return 0.0, []
            best_match = max(matches, key=lambda prediction: prediction.confidence_score)
            explanations.append(
                f"Candidate is predicted to interact with the specified "
                f"target of interest ('{query.query_target.name}') with "
                f"confidence {best_match.confidence_score:.2f}."
            )
            return best_match.confidence_score, matches

        if not query_target_predictions:
            explanations.append(
                "Target overlap not evaluated: no predicted targets are "
                "available for the query."
            )
            return 0.0, []

        query_targets_by_id = {
            prediction.target.target_id: prediction
            for prediction in query_target_predictions
        }
        shared: list[TargetPrediction] = []
        overlap_contributions: list[float] = []
        for candidate_prediction in candidate_target_predictions:
            query_prediction = query_targets_by_id.get(
                candidate_prediction.target.target_id
            )
            if query_prediction is not None:
                shared.append(candidate_prediction)
                overlap_contributions.append(
                    min(
                        candidate_prediction.confidence_score,
                        query_prediction.confidence_score,
                    )
                )

        if not shared:
            explanations.append(
                "No predicted targets are shared between the candidate "
                "and the query compound."
            )
            return 0.0, []

        score = min(1.0, sum(overlap_contributions) / len(query_targets_by_id))
        shared_names = ", ".join(prediction.target.name for prediction in shared)
        explanations.append(
            f"Candidate shares {len(shared)} predicted target(s) with the "
            f"query compound: {shared_names}."
        )
        return score, shared

    @staticmethod
    def _score_embedding_similarity(
        query: RepurposingQuery,
        candidate: Compound,
        embedding_provider: EmbeddingSimilarityProvider | None,
        explanations: list[str],
    ) -> float:
        """
        Compute the semantic embedding similarity contribution for a
        candidate.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate: The candidate ``Compound`` being scored.
            embedding_provider: The injected embedding similarity
                provider, or None if unavailable.
            explanations: List to which a human-readable explanation of
                this score is appended, in place.

        Returns:
            The embedding similarity score, in [0.0, 1.0].
        """
        if embedding_provider is None:
            explanations.append(
                "Embedding similarity not evaluated: no embedding "
                "provider was supplied."
            )
            return 0.0

        if query.disease_name:
            query_entity_id, query_entity_type = (
                query.disease_name,
                KnowledgeGraphEntityType.DISEASE.value,
            )
        elif query.effective_compound_id:
            query_entity_id, query_entity_type = (
                query.effective_compound_id,
                KnowledgeGraphEntityType.DRUG.value,
            )
        else:
            explanations.append(
                "Embedding similarity not evaluated: the query does not "
                "carry a disease name or compound identifier suitable for "
                "embedding lookup."
            )
            return 0.0

        try:
            score = embedding_provider.get_similarity(
                query_entity_id,
                query_entity_type,
                candidate.compound_id,
                KnowledgeGraphEntityType.DRUG.value,
            )
        except Exception:  # noqa: BLE001 - degrade gracefully, do not fail scoring
            logger.exception(
                "Embedding similarity lookup failed for candidate '%s'; "
                "treating as unavailable.",
                candidate.compound_id,
            )
            explanations.append(
                "Embedding similarity could not be retrieved due to an "
                "unexpected error and was treated as unavailable."
            )
            return 0.0

        score = min(1.0, max(0.0, score))
        explanations.append(
            f"Semantic embedding similarity between '{query_entity_id}' and "
            f"the candidate is {score:.2f}."
        )
        return score

    @staticmethod
    def _score_knowledge_graph(
        query: RepurposingQuery,
        candidate: Compound,
        kg_provider: KnowledgeGraphProvider | None,
        explanations: list[str],
    ) -> tuple[float, list[KnowledgeGraphRelationship]]:
        """
        Compute the Knowledge Graph relationship contribution for a
        candidate.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate: The candidate ``Compound`` being scored.
            kg_provider: The injected Knowledge Graph provider, or None
                if unavailable.
            explanations: List to which a human-readable explanation of
                this score is appended, in place.

        Returns:
            A tuple of the Knowledge Graph score, in [0.0, 1.0], and the
            list of contributing ``KnowledgeGraphRelationship`` instances.
        """
        if kg_provider is None:
            explanations.append(
                "Knowledge Graph evidence not evaluated: no Knowledge "
                "Graph provider was supplied."
            )
            return 0.0, []

        if not query.disease_name:
            explanations.append(
                "Knowledge Graph evidence not evaluated: no disease name "
                "was provided in the query."
            )
            return 0.0, []

        try:
            relationships = kg_provider.get_relationships(
                candidate.compound_id,
                KnowledgeGraphEntityType.DRUG.value,
                query.disease_name,
                KnowledgeGraphEntityType.DISEASE.value,
            )
        except Exception:  # noqa: BLE001 - degrade gracefully, do not fail scoring
            logger.exception(
                "Knowledge Graph relationship lookup failed for candidate "
                "'%s'; treating as unavailable.",
                candidate.compound_id,
            )
            explanations.append(
                "Knowledge Graph relationships could not be retrieved due "
                "to an unexpected error and were treated as unavailable."
            )
            return 0.0, []

        if not relationships:
            explanations.append(
                f"No Knowledge Graph relationships were found connecting "
                f"this candidate to '{query.disease_name}'."
            )
            return 0.0, []

        score = min(1.0, max(r.strength for r in relationships))
        path_descriptions = "; ".join(r.path_description for r in relationships)
        explanations.append(
            f"{len(relationships)} Knowledge Graph relationship(s) connect "
            f"this candidate to '{query.disease_name}': {path_descriptions}."
        )
        return score, relationships

    def score_candidate(
        self,
        query: RepurposingQuery,
        candidate: Compound,
        candidate_target_predictions: list[TargetPrediction],
        query_target_predictions: list[TargetPrediction],
        embedding_provider: EmbeddingSimilarityProvider | None = None,
        kg_provider: KnowledgeGraphProvider | None = None,
    ) -> RepurposingEvidence:
        """
        Score a single candidate compound's suitability for repurposing
        against the given query, combining all four evidence sources.

        Args:
            query: The normalized ``RepurposingQuery``.
            candidate: The candidate ``Compound`` being scored.
            candidate_target_predictions: Predicted targets for the
                candidate compound.
            query_target_predictions: Predicted targets for the query
                compound (empty if unavailable).
            embedding_provider: Optional embedding similarity provider.
            kg_provider: Optional Knowledge Graph provider.

        Returns:
            A populated ``RepurposingEvidence`` instance.

        Raises:
            RepurposingScoringError: If scoring fails unexpectedly.
        """
        explanations: list[str] = []

        try:
            structural_score = self._score_structural_similarity(
                query, candidate, explanations
            )
            target_overlap_score, shared_targets = self._score_target_overlap(
                query,
                candidate_target_predictions,
                query_target_predictions,
                explanations,
            )
            embedding_score = self._score_embedding_similarity(
                query, candidate, embedding_provider, explanations
            )
            kg_score, kg_relationships = self._score_knowledge_graph(
                query, candidate, kg_provider, explanations
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Repurposing scoring failed unexpectedly for candidate "
                "'%s'.",
                candidate.compound_id,
            )
            raise RepurposingScoringError(
                candidate_id=candidate.compound_id, engine_name=self.engine_name
            ) from exc

        component_scores = {
            "structural_similarity_score": structural_score,
            "target_overlap_score": target_overlap_score,
            "embedding_similarity_score": embedding_score,
            "knowledge_graph_score": kg_score,
        }
        composite_score = min(
            1.0,
            sum(
                component_scores[source] * weight
                for source, weight in self.weights.items()
                if source in component_scores
            ),
        )
        confidence = self._score_to_confidence(composite_score)

        explanations.insert(
            0,
            f"Composite repurposing score is {composite_score:.2f} "
            f"(confidence: {confidence.value}).",
        )

        return RepurposingEvidence(
            candidate_id=candidate.compound_id,
            structural_similarity_score=structural_score,
            target_overlap_score=target_overlap_score,
            embedding_similarity_score=embedding_score,
            knowledge_graph_score=kg_score,
            composite_score=composite_score,
            confidence=confidence,
            shared_targets=shared_targets,
            knowledge_graph_relationships=kg_relationships,
            explanations=explanations,
        )


# ---------------------------------------------------------------------------
# Repurposing service (orchestration layer)
# ---------------------------------------------------------------------------


class DrugRepurposingService:
    """
    Orchestrates query normalization, target prediction, and multi-source
    evidence scoring to identify and rank drug repurposing candidates.

    This is the primary integration point for ``pipeline.py``. The
    scoring backend can be swapped (e.g., for a graph neural network or
    other AI-driven repurposing model) by constructing this service with
    a different ``RepurposingScoringEngine`` implementation, without
    changing any calling code. Similarly, Embeddings and Knowledge Graph
    integrations are supplied as adapter objects satisfying
    ``EmbeddingSimilarityProvider`` and ``KnowledgeGraphProvider``,
    keeping this module decoupled from those modules' internals.

    Attributes:
        scoring_engine: The active ``RepurposingScoringEngine`` used for
            candidate scoring.
        target_prediction_service: The ``TargetPredictionService`` used
            to predict biological targets for the query and candidates.
        embedding_provider: Optional adapter over the Embeddings module.
        kg_provider: Optional adapter over the Knowledge Graph module.
    """

    def __init__(
        self,
        scoring_engine: RepurposingScoringEngine | None = None,
        target_prediction_service: TargetPredictionService | None = None,
        embedding_provider: EmbeddingSimilarityProvider | None = None,
        kg_provider: KnowledgeGraphProvider | None = None,
    ) -> None:
        """
        Initialize the drug repurposing service.

        Args:
            scoring_engine: The candidate scoring engine to use. If
                omitted, a ``RuleBasedRepurposingScorer`` baseline is
                used.
            target_prediction_service: The target prediction service to
                use for both the query and candidate compounds. If
                omitted, a default ``TargetPredictionService`` (rule-based
                baseline) is constructed.
            embedding_provider: Optional adapter over BioNexus's
                Embeddings module, satisfying
                ``EmbeddingSimilarityProvider``. If omitted, embedding
                similarity contributes zero to every candidate's score.
            kg_provider: Optional adapter over BioNexus's Knowledge Graph
                module, satisfying ``KnowledgeGraphProvider``. If
                omitted, Knowledge Graph evidence contributes zero to
                every candidate's score.
        """
        self.scoring_engine = scoring_engine or RuleBasedRepurposingScorer()
        self.target_prediction_service = (
            target_prediction_service or TargetPredictionService()
        )
        self.embedding_provider = embedding_provider
        self.kg_provider = kg_provider

    def _predict_targets_safely(
        self, smiles: str | None, compound_id: str, warnings: list[str]
    ) -> list[TargetPrediction]:
        """
        Predict targets for a compound, degrading gracefully on failure.

        Args:
            smiles: SMILES string of the compound, or None if
                unavailable.
            compound_id: Identifier of the compound.
            warnings: List to which a warning message is appended, in
                place, if prediction could not be performed.

        Returns:
            A list of ``TargetPrediction`` instances, or an empty list if
            ``smiles`` is None or prediction failed.
        """
        if not smiles:
            return []
        try:
            return self.target_prediction_service.predict_targets(
                smiles=smiles, compound_id=compound_id
            )
        except InvalidSMILESError:
            message = (
                f"Skipped target prediction for '{compound_id}': invalid "
                f"SMILES string."
            )
            logger.warning(message)
            warnings.append(message)
            return []
        except DescriptorCalculationError as exc:
            message = (
                f"Skipped target prediction for '{compound_id}' due to a "
                f"descriptor calculation failure: {exc}"
            )
            logger.warning(message)
            warnings.append(message)
            return []

    def find_repurposing_candidates(
        self,
        candidates: list[Compound],
        disease_name: str | None = None,
        query_compound: Compound | None = None,
        query_smiles: str | None = None,
        query_target: DrugTarget | None = None,
        top_k: int | None = None,
        min_confidence: float | None = None,
        skip_invalid_candidates: bool = True,
    ) -> DrugDiscoveryResult:
        """
        Identify and rank drug repurposing candidates against a query.

        The query may be specified as any combination of a disease name,
        a reference compound, a raw reference SMILES string, and/or a
        specific biological target of interest.

        Args:
            candidates: The pool of candidate ``Compound`` instances to
                evaluate.
            disease_name: Name of the disease being targeted for
                repurposing.
            query_compound: A reference ``Compound`` to compare
                candidates against.
            query_smiles: A reference SMILES string, used when only a raw
                structure is available.
            query_target: A specific ``DrugTarget`` of interest.
            top_k: If provided, limits the number of ranked candidates
                returned to the top ``top_k`` highest composite scores.
            min_confidence: If provided, filters out candidates whose
                composite score falls below this threshold, in
                [0.0, 1.0].
            skip_invalid_candidates: If True, candidates with unparseable
                SMILES are logged as warnings and still scored using
                whatever evidence remains available (structural
                similarity and target overlap will contribute 0.0 for
                such candidates). If False, an invalid candidate SMILES
                raises ``InvalidSMILESError``.

        Returns:
            A ``DrugDiscoveryResult`` containing one ``DrugCandidate`` and
            one ``CandidateRanking`` per evaluated candidate, sorted from
            best to worst by composite score.

        Raises:
            RepurposingQueryError: If no disease name, compound, SMILES,
                or target was provided.
            InvalidSMILESError: If ``query_smiles`` or
                ``query_compound.smiles`` cannot be parsed, or if a
                candidate's SMILES cannot be parsed and
                ``skip_invalid_candidates`` is False.
            RepurposingScoringError: If the underlying scoring engine
                fails unexpectedly for a candidate.
            ValueError: If ``top_k`` or ``min_confidence`` are out of
                valid range.
        """
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be a positive integer when provided.")
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0.0, 1.0].")

        query = RepurposingQuery(
            disease_name=disease_name,
            query_compound=query_compound,
            query_smiles=query_smiles,
            query_target=query_target,
        )

        run_id = str(uuid.uuid4())
        warnings: list[str] = []

        logger.info(
            "Starting drug repurposing search run_id='%s' against %d "
            "candidate(s) using scoring engine '%s'.",
            run_id,
            len(candidates),
            self.scoring_engine.engine_name,
        )

        query_target_predictions = self._predict_targets_safely(
            smiles=query.effective_smiles,
            compound_id=query.effective_compound_id or "query_compound",
            warnings=warnings,
        )

        drug_candidates: list[DrugCandidate] = []
        evidence_by_candidate_id: dict[str, RepurposingEvidence] = {}

        for candidate in candidates:
            candidate_target_predictions = self._predict_targets_safely(
                smiles=candidate.smiles,
                compound_id=candidate.compound_id,
                warnings=warnings,
            )

            try:
                evidence = self.scoring_engine.score_candidate(
                    query=query,
                    candidate=candidate,
                    candidate_target_predictions=candidate_target_predictions,
                    query_target_predictions=query_target_predictions,
                    **(
                        {
                            "embedding_provider": self.embedding_provider,
                            "kg_provider": self.kg_provider,
                        }
                        if isinstance(self.scoring_engine, RuleBasedRepurposingScorer)
                        else {}
                    ),
                )
            except RepurposingScoringError:
                if skip_invalid_candidates:
                    message = (
                        f"Skipped scoring for candidate "
                        f"'{candidate.compound_id}' due to a scoring "
                        f"engine failure."
                    )
                    logger.warning(message)
                    warnings.append(message)
                    continue
                raise

            evidence_by_candidate_id[candidate.compound_id] = evidence

            if min_confidence is not None and evidence.composite_score < min_confidence:
                continue

            drug_candidates.append(
                DrugCandidate(
                    candidate_id=f"REPURPOSE_{candidate.compound_id}",
                    compound=candidate,
                    target_predictions=candidate_target_predictions,
                    status=CandidateStatus.PENDING,
                    overall_score=evidence.composite_score,
                    notes=" ".join(evidence.explanations),
                )
            )

        drug_candidates.sort(
            key=lambda dc: dc.overall_score if dc.overall_score is not None else 0.0,
            reverse=True,
        )

        if top_k is not None:
            drug_candidates = drug_candidates[:top_k]

        rankings: list[CandidateRanking] = []
        for rank, candidate_entry in enumerate(drug_candidates, start=1):
            evidence = evidence_by_candidate_id[candidate_entry.compound.compound_id]
            rankings.append(
                CandidateRanking(
                    candidate_id=candidate_entry.candidate_id,
                    rank=rank,
                    composite_score=evidence.composite_score,
                    criteria_scores={
                        "structural_similarity": evidence.structural_similarity_score,
                        "target_overlap": evidence.target_overlap_score,
                        "embedding_similarity": evidence.embedding_similarity_score,
                        "knowledge_graph": evidence.knowledge_graph_score,
                    },
                    notes=(
                        f"Confidence: {evidence.confidence.value}. "
                        + " ".join(evidence.explanations)
                    ),
                )
            )

        result = DrugDiscoveryResult(
            run_id=run_id,
            query_compound=query.query_compound,
            candidates=drug_candidates,
            rankings=rankings,
            stage=DiscoveryStage.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            warnings=warnings,
            metadata={
                "disease_name": disease_name,
                "query_smiles": query.effective_smiles,
                "query_target_id": (
                    query_target.target_id if query_target is not None else None
                ),
                "scoring_engine": self.scoring_engine.engine_name,
                "candidate_pool_size": len(candidates),
                "ranked_candidate_count": len(drug_candidates),
            },
        )

        logger.info(
            "Completed drug repurposing search run_id='%s': %d candidate(s) "
            "ranked.",
            run_id,
            len(drug_candidates),
        )
        return result


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def find_repurposing_candidates(
    candidates: list[Compound],
    disease_name: str | None = None,
    query_compound: Compound | None = None,
    query_smiles: str | None = None,
    query_target: DrugTarget | None = None,
    top_k: int | None = None,
    min_confidence: float | None = None,
    embedding_provider: EmbeddingSimilarityProvider | None = None,
    kg_provider: KnowledgeGraphProvider | None = None,
) -> DrugDiscoveryResult:
    """
    Identify and rank drug repurposing candidates against a query, using
    a default rule-based ``DrugRepurposingService``.

    This is the primary entry point intended for use by ``pipeline.py``.

    Args:
        candidates: The pool of candidate ``Compound`` instances to
            evaluate.
        disease_name: Name of the disease being targeted for
            repurposing.
        query_compound: A reference ``Compound`` to compare candidates
            against.
        query_smiles: A reference SMILES string, used when only a raw
            structure is available.
        query_target: A specific ``DrugTarget`` of interest.
        top_k: If provided, limits the number of ranked candidates
            returned to the top ``top_k`` highest composite scores.
        min_confidence: If provided, filters out candidates whose
            composite score falls below this threshold, in [0.0, 1.0].
        embedding_provider: Optional adapter over BioNexus's Embeddings
            module.
        kg_provider: Optional adapter over BioNexus's Knowledge Graph
            module.

    Returns:
        A ``DrugDiscoveryResult`` containing ranked repurposing
        candidates, from best to worst.

    Raises:
        RepurposingQueryError: If no disease name, compound, SMILES, or
            target was provided.
        InvalidSMILESError: If a supplied SMILES string cannot be parsed.
        RepurposingScoringError: If the underlying scoring engine fails
            unexpectedly for a candidate.
    """
    service = DrugRepurposingService(
        embedding_provider=embedding_provider, kg_provider=kg_provider
    )
    return service.find_repurposing_candidates(
        candidates=candidates,
        disease_name=disease_name,
        query_compound=query_compound,
        query_smiles=query_smiles,
        query_target=query_target,
        top_k=top_k,
        min_confidence=min_confidence,
    )