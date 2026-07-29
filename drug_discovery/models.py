"""
drug_discovery/models.py

Data models for the BioNexus Drug Discovery module.

This module defines the core data structures used throughout the Drug
Discovery pipeline, including compounds, molecular properties, target
predictions, toxicity assessments, candidate ranking, and pipeline
configuration.

Design notes
------------
- This module contains ONLY data models (dataclasses), enums, and
  configuration containers. No business logic, scoring algorithms, or
  I/O operations are implemented here.
- Validation is limited to structural/range checks on the data itself
  (e.g., scores must be within [0, 1]), performed in ``__post_init__``.
- All models are designed to be consumed by downstream modules such as
  ``compound_similarity.py``, ``target_prediction.py``,
  ``drug_repurposing.py``, and ``pipeline.py``.
- Models are intentionally immutable-friendly (fields are simple,
  serializable types) to support caching, hashing, and use within a
  Streamlit session state without unexpected mutation side effects.

Compatibility
-------------
Targets Python 3.11 and follows the same architectural conventions used
by the existing BioNexus modules (Literature Search, Drug Info,
Knowledge Graph, SMILES Analyzer, Embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TargetType(str, Enum):
    """Classification of a biological drug target."""

    PROTEIN = "protein"
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    ION_CHANNEL = "ion_channel"
    TRANSPORTER = "transporter"
    NUCLEIC_ACID = "nucleic_acid"
    OTHER = "other"


class SimilarityMetric(str, Enum):
    """Molecular similarity metrics supported by the platform."""

    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class PredictionConfidence(str, Enum):
    """Qualitative confidence bucket associated with a model prediction."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ToxicityLevel(str, Enum):
    """Qualitative toxicity risk classification for a compound."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class CandidateStatus(str, Enum):
    """Lifecycle status of a drug candidate within the discovery pipeline."""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    PRIORITIZED = "prioritized"
    REJECTED = "rejected"


class DiscoveryStage(str, Enum):
    """Stages of the drug discovery pipeline, used for run metadata."""

    SIMILARITY_SEARCH = "similarity_search"
    TARGET_PREDICTION = "target_prediction"
    TOXICITY_PREDICTION = "toxicity_prediction"
    CANDIDATE_RANKING = "candidate_ranking"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Core compound models
# ---------------------------------------------------------------------------


@dataclass
class Compound:
    """
    Represents a chemical compound within the BioNexus platform.

    Attributes:
        compound_id: Unique identifier for the compound (e.g., internal ID,
            PubChem CID, or ChEMBL ID).
        name: Human-readable name of the compound.
        smiles: Canonical SMILES string representing the compound structure.
        inchi: InChI string, if available.
        inchi_key: InChIKey hash, if available.
        molecular_formula: Molecular formula (e.g., "C9H8O4").
        molecular_weight: Molecular weight in g/mol.
        source: Origin of the compound record (e.g., "PubChem", "ChEMBL",
            "user_upload").
        metadata: Arbitrary additional attributes not otherwise modeled.
    """

    compound_id: str
    name: str
    smiles: str
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate structural invariants of the compound record."""
        if not self.compound_id.strip():
            raise ValueError("compound_id must be a non-empty string.")
        if not self.smiles.strip():
            raise ValueError("smiles must be a non-empty string.")
        if self.molecular_weight is not None and self.molecular_weight < 0:
            raise ValueError("molecular_weight cannot be negative.")


@dataclass
class SimilarCompound:
    """
    Represents a compound returned from a similarity search, paired with
    its similarity score relative to a query compound.

    Attributes:
        compound: The matched ``Compound`` instance.
        similarity_score: Similarity score in the range [0.0, 1.0].
        metric: The similarity metric used to compute the score.
        rank: Rank position (1-based) of this result within its result set.
    """

    compound: Compound
    similarity_score: float
    metric: SimilarityMetric
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the similarity score and rank."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"similarity_score must be within [0.0, 1.0], "
                f"got {self.similarity_score}."
            )
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be a positive integer when provided.")


@dataclass
class MolecularProperties:
    """
    Computed physicochemical properties of a compound.

    Attributes:
        compound_id: Identifier of the ``Compound`` these properties belong
            to.
        molecular_weight: Molecular weight in g/mol.
        logp: Calculated octanol-water partition coefficient (LogP).
        tpsa: Topological polar surface area (angstrom^2).
        h_bond_donors: Count of hydrogen bond donors.
        h_bond_acceptors: Count of hydrogen bond acceptors.
        rotatable_bonds: Count of rotatable bonds.
        aromatic_rings: Count of aromatic rings.
        heavy_atom_count: Count of non-hydrogen atoms.
        lipinski_violations: Number of Lipinski's Rule of Five violations.
        qed_score: Quantitative Estimate of Drug-likeness, in [0.0, 1.0].
    """

    compound_id: str
    molecular_weight: float
    logp: float
    tpsa: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atom_count: int
    lipinski_violations: int
    qed_score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate that computed property values fall within sane ranges."""
        non_negative_fields: dict[str, float] = {
            "molecular_weight": self.molecular_weight,
            "tpsa": self.tpsa,
            "h_bond_donors": self.h_bond_donors,
            "h_bond_acceptors": self.h_bond_acceptors,
            "rotatable_bonds": self.rotatable_bonds,
            "aromatic_rings": self.aromatic_rings,
            "heavy_atom_count": self.heavy_atom_count,
            "lipinski_violations": self.lipinski_violations,
        }
        for field_name, value in non_negative_fields.items():
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative.")
        if self.qed_score is not None and not 0.0 <= self.qed_score <= 1.0:
            raise ValueError("qed_score must be within [0.0, 1.0].")

    def is_lipinski_compliant(self) -> bool:
        """
        Return whether the compound satisfies Lipinski's Rule of Five.

        Returns:
            True if there are zero Lipinski violations, False otherwise.
        """
        return self.lipinski_violations == 0


# ---------------------------------------------------------------------------
# Target and prediction models
# ---------------------------------------------------------------------------


@dataclass
class DrugTarget:
    """
    Represents a biological target (e.g., protein, enzyme, receptor).

    Attributes:
        target_id: Unique identifier for the target (e.g., internal ID).
        name: Human-readable target name.
        target_type: Classification of the target.
        organism: Source organism (e.g., "Homo sapiens").
        uniprot_id: UniProt accession identifier, if available.
        gene_symbol: Associated gene symbol, if available.
        description: Free-text description of the target.
    """

    target_id: str
    name: str
    target_type: TargetType
    organism: Optional[str] = None
    uniprot_id: Optional[str] = None
    gene_symbol: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate required identifying fields."""
        if not self.target_id.strip():
            raise ValueError("target_id must be a non-empty string.")
        if not self.name.strip():
            raise ValueError("name must be a non-empty string.")


@dataclass
class TargetPrediction:
    """
    Represents a predicted interaction between a compound and a target.

    Attributes:
        compound_id: Identifier of the compound being evaluated.
        target: The ``DrugTarget`` predicted to interact with the compound.
        confidence_score: Numeric confidence score in [0.0, 1.0].
        confidence_level: Qualitative confidence bucket.
        binding_affinity: Predicted binding affinity value (e.g., pKi, pIC50),
            if available.
        binding_affinity_unit: Unit of the reported binding affinity
            (e.g., "pIC50", "nM").
        prediction_method: Name/identifier of the model or method used to
            generate the prediction.
    """

    compound_id: str
    target: DrugTarget
    confidence_score: float
    confidence_level: PredictionConfidence
    binding_affinity: Optional[float] = None
    binding_affinity_unit: Optional[str] = None
    prediction_method: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the confidence score range."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be within [0.0, 1.0], "
                f"got {self.confidence_score}."
            )


@dataclass
class ToxicityPrediction:
    """
    Represents a predicted toxicity profile for a compound.

    Attributes:
        compound_id: Identifier of the compound being evaluated.
        toxicity_level: Overall qualitative toxicity classification.
        toxicity_score: Overall numeric toxicity score in [0.0, 1.0], where
            higher values indicate greater predicted toxicity risk.
        endpoint_scores: Mapping of specific toxicity endpoints (e.g.,
            "hepatotoxicity", "cardiotoxicity", "mutagenicity") to their
            individual numeric scores in [0.0, 1.0].
        confidence: Qualitative confidence bucket for the prediction.
        prediction_method: Name/identifier of the model or method used to
            generate the prediction.
    """

    compound_id: str
    toxicity_level: ToxicityLevel
    toxicity_score: float
    endpoint_scores: dict[str, float] = field(default_factory=dict)
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM
    prediction_method: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate toxicity score ranges for the overall and per-endpoint
        scores."""
        if not 0.0 <= self.toxicity_score <= 1.0:
            raise ValueError(
                f"toxicity_score must be within [0.0, 1.0], "
                f"got {self.toxicity_score}."
            )
        for endpoint, score in self.endpoint_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"endpoint_scores['{endpoint}'] must be within "
                    f"[0.0, 1.0], got {score}."
                )


# ---------------------------------------------------------------------------
# Candidate and ranking models
# ---------------------------------------------------------------------------


@dataclass
class DrugCandidate:
    """
    Represents a candidate compound produced by the drug discovery pipeline,
    aggregating its structural, target, and toxicity information.

    Attributes:
        candidate_id: Unique identifier for this candidate record.
        compound: The underlying ``Compound``.
        molecular_properties: Computed physicochemical properties, if
            available.
        target_predictions: List of predicted target interactions.
        toxicity_prediction: Predicted toxicity profile, if available.
        status: Current lifecycle status of the candidate.
        overall_score: Optional composite score summarizing the candidate's
            overall desirability, in [0.0, 1.0].
        notes: Free-text annotations or reviewer comments.
    """

    candidate_id: str
    compound: Compound
    molecular_properties: Optional[MolecularProperties] = None
    target_predictions: list[TargetPrediction] = field(default_factory=list)
    toxicity_prediction: Optional[ToxicityPrediction] = None
    status: CandidateStatus = CandidateStatus.PENDING
    overall_score: Optional[float] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate identifiers and the optional overall score range."""
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if self.overall_score is not None and not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("overall_score must be within [0.0, 1.0].")

    def top_target_prediction(self) -> Optional[TargetPrediction]:
        """
        Return the target prediction with the highest confidence score.

        Returns:
            The ``TargetPrediction`` with the highest ``confidence_score``,
            or None if no target predictions are present.
        """
        if not self.target_predictions:
            return None
        return max(self.target_predictions, key=lambda tp: tp.confidence_score)


@dataclass
class CandidateRanking:
    """
    Represents the ranking outcome for a single drug candidate within a
    completed discovery run.

    Attributes:
        candidate_id: Identifier of the ranked ``DrugCandidate``.
        rank: Rank position (1-based, lower is better) within the run.
        composite_score: Final composite score used to determine rank, in
            [0.0, 1.0].
        criteria_scores: Mapping of individual ranking criteria (e.g.,
            "efficacy", "safety", "novelty") to their contributing scores.
        notes: Free-text notes explaining the ranking rationale.
    """

    candidate_id: str
    rank: int
    composite_score: float
    criteria_scores: dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate rank and composite score values."""
        if self.rank < 1:
            raise ValueError("rank must be a positive integer.")
        if not 0.0 <= self.composite_score <= 1.0:
            raise ValueError("composite_score must be within [0.0, 1.0].")
        for criterion, score in self.criteria_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"criteria_scores['{criterion}'] must be within "
                    f"[0.0, 1.0], got {score}."
                )


# ---------------------------------------------------------------------------
# Pipeline result and configuration models
# ---------------------------------------------------------------------------


@dataclass
class DrugDiscoveryResult:
    """
    Represents the complete output of a drug discovery pipeline run.

    Attributes:
        run_id: Unique identifier for this pipeline execution.
        query_compound: The compound (or seed compound) that initiated the
            discovery run, if applicable.
        candidates: List of drug candidates produced by the run.
        rankings: List of ranking outcomes corresponding to the candidates.
        stage: The current or final stage of the pipeline for this run.
        created_at: UTC timestamp marking when the run was created.
        completed_at: UTC timestamp marking when the run completed, if
            finished.
        warnings: Non-fatal warnings collected during the run.
        metadata: Arbitrary additional run metadata (e.g., configuration
            snapshot, execution duration).
    """

    run_id: str
    query_compound: Optional[Compound] = None
    candidates: list[DrugCandidate] = field(default_factory=list)
    rankings: list[CandidateRanking] = field(default_factory=list)
    stage: DiscoveryStage = DiscoveryStage.SIMILARITY_SEARCH
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and timestamp ordering."""
        if not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot be earlier than created_at.")

    def is_completed(self) -> bool:
        """
        Return whether this discovery run has completed.

        Returns:
            True if the stage is ``DiscoveryStage.COMPLETED``, False
            otherwise.
        """
        return self.stage == DiscoveryStage.COMPLETED

    def ranked_candidates(self) -> list[DrugCandidate]:
        """
        Return candidates ordered according to their associated ranking.

        Candidates without a corresponding entry in ``rankings`` are
        appended after ranked candidates, in their original order.

        Returns:
            A list of ``DrugCandidate`` instances ordered by rank ascending.
        """
        rank_by_id = {r.candidate_id: r.rank for r in self.rankings}
        ranked = [c for c in self.candidates if c.candidate_id in rank_by_id]
        unranked = [c for c in self.candidates if c.candidate_id not in rank_by_id]
        ranked.sort(key=lambda c: rank_by_id[c.candidate_id])
        return ranked + unranked


@dataclass
class DrugDiscoveryConfig:
    """
    Configuration parameters governing a drug discovery pipeline run.

    Attributes:
        similarity_metric: Similarity metric to use for compound search.
        similarity_threshold: Minimum similarity score, in [0.0, 1.0],
            required for a candidate to be retained during similarity
            search.
        max_similar_compounds: Maximum number of similar compounds to
            retrieve per query.
        max_targets_per_compound: Maximum number of target predictions to
            retain per compound.
        min_target_confidence: Minimum confidence score, in [0.0, 1.0],
            required for a target prediction to be retained.
        toxicity_score_threshold: Maximum acceptable toxicity score, in
            [0.0, 1.0], above which a candidate is flagged or excluded.
        ranking_weights: Mapping of ranking criteria names to their relative
            weights, used when computing a composite score. Weights should
            sum to approximately 1.0 but this is not strictly enforced here.
        enable_toxicity_prediction: Whether toxicity prediction is enabled
            for the run.
        enable_target_prediction: Whether target prediction is enabled for
            the run.
        random_seed: Optional seed for reproducibility of stochastic steps
            in downstream algorithms.
    """

    similarity_metric: SimilarityMetric = SimilarityMetric.TANIMOTO
    similarity_threshold: float = 0.7
    max_similar_compounds: int = 25
    max_targets_per_compound: int = 10
    min_target_confidence: float = 0.5
    toxicity_score_threshold: float = 0.6
    ranking_weights: dict[str, float] = field(default_factory=dict)
    enable_toxicity_prediction: bool = True
    enable_target_prediction: bool = True
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate threshold ranges and configuration limits."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [0.0, 1.0].")
        if not 0.0 <= self.min_target_confidence <= 1.0:
            raise ValueError("min_target_confidence must be within [0.0, 1.0].")
        if not 0.0 <= self.toxicity_score_threshold <= 1.0:
            raise ValueError(
                "toxicity_score_threshold must be within [0.0, 1.0]."
            )
        if self.max_similar_compounds < 1:
            raise ValueError("max_similar_compounds must be a positive integer.")
        if self.max_targets_per_compound < 1:
            raise ValueError(
                "max_targets_per_compound must be a positive integer."
            )
        for criterion, weight in self.ranking_weights.items():
            if weight < 0:
                raise ValueError(
                    f"ranking_weights['{criterion}'] cannot be negative."
                )