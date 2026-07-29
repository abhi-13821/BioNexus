"""
drug_discovery/pipeline.py

Central orchestration layer for the BioNexus Drug Discovery module.

This module coordinates all Drug Discovery components into a complete,
production-ready workflow. It accepts compound inputs, orchestrates the
execution of molecular property calculation, similarity search, target
prediction, toxicity prediction, drug repurposing, and candidate ranking,
while providing comprehensive logging, caching, error handling, and
execution metadata collection.

The pipeline is designed to be:
    - Modular: Each stage is a discrete, well-defined step.
    - Extensible: New stages or alternative backends can be added easily.
    - Robust: Partial failures are handled gracefully where possible.
    - Observable: Detailed logging and timing information is collected.
    - Cache-aware: Leverages the caching layer to avoid redundant computation.

Integration Points:
    - molecular_properties: calculate_molecular_properties
    - compound_similarity: find_similar_compounds
    - target_prediction: predict_targets
    - toxicity_prediction: predict_toxicity
    - drug_repurposing: find_repurposing_candidates
    - candidate_ranking: rank_candidates
    - cache: get_default_cache, cached decorators
    - utils: timing, validation, statistics, formatting

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from drug_discovery.cache import (
    CacheNamespace,
    cache_candidate_ranking,
    cache_compound_similarity,
    cache_drug_repurposing,
    cache_molecular_properties,
    cache_target_prediction,
    cache_toxicity_prediction,
    get_default_cache,
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
from drug_discovery.compound_similarity import (
    CompoundSimilaritySearchEngine,
    FingerprintConfig,
    find_similar_compounds,
)
from drug_discovery.drug_repurposing import (
    DrugRepurposingService,
    EmbeddingSimilarityProvider,
    KnowledgeGraphProvider,
    RepurposingScoringError,
    RuleBasedRepurposingScorer,
    find_repurposing_candidates,
)
from drug_discovery.models import (
    CandidateRanking,
    Compound,
    DiscoveryStage,
    DrugCandidate,
    DrugDiscoveryConfig,
    DrugDiscoveryResult,
    MolecularProperties,
    TargetPrediction,
    ToxicityPrediction,
)
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
    MolecularPropertyCalculator,
    calculate_molecular_properties,
)
from drug_discovery.target_prediction import (
    TargetPredictionError,
    TargetPredictionService,
    predict_targets,
)
from drug_discovery.toxicity_prediction import (
    ToxicityPredictionError,
    ToxicityPredictionService,
    predict_toxicity,
)
from drug_discovery.utils import (
    InputValidationError,
    compute_basic_statistics,
    format_iso_timestamp,
    generate_unique_id,
    safe_mean,
    timed,
    utc_now,
    validate_non_empty_string,
    validate_positive_integer,
    validate_probability,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Base exception for pipeline-level failures."""

    pass


class PipelineConfigurationError(ValueError):
    """
    Raised when the pipeline is configured with invalid parameters.
    """

    pass


class PipelineStageError(PipelineError):
    """
    Raised when a specific pipeline stage fails, preserving the context of
    which stage failed and the original exception.

    Attributes:
        stage_name: Name of the stage that failed.
        original_exception: The exception that caused the failure.
    """

    def __init__(
        self,
        stage_name: str,
        original_exception: Exception,
        message: Optional[str] = None,
    ) -> None:
        self.stage_name = stage_name
        self.original_exception = original_exception
        resolved_message = (
            message
            or f"Pipeline stage '{stage_name}' failed: {original_exception}"
        )
        super().__init__(resolved_message)


# ----------------------------------------------------------------------
# Pipeline configuration
# ----------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Configuration for a drug discovery pipeline run.

    Attributes:
        discovery_config: Core drug discovery configuration parameters.
        fingerprint_config: Configuration for Morgan fingerprint generation.
        ranking_config: Configuration for candidate ranking.
        ranking_strategy: Strategy to use for combining ranking criteria.
        ranking_filter: Filter to apply to candidates before final ranking.
        top_k: Maximum number of candidates to return after ranking.
        min_ranking_confidence: Minimum confidence score for a candidate
            to be included in the final ranking.
        enable_cache: Whether to use the caching layer.
        enable_similarity: Whether to perform similarity search.
        enable_target_prediction: Whether to perform target prediction.
        enable_toxicity_prediction: Whether to perform toxicity prediction.
        enable_repurposing: Whether to perform drug repurposing analysis.
        embedding_provider: Optional embedding similarity provider for
            repurposing analysis.
        kg_provider: Optional knowledge graph provider for repurposing
            analysis.
        skip_invalid_compounds: If True, compounds with invalid SMILES
            are skipped rather than causing the entire pipeline to fail.
        max_compound_errors: Maximum number of compound processing errors
            before aborting the pipeline.
        collect_timing: Whether to collect per-stage timing information.
    """

    discovery_config: DrugDiscoveryConfig = field(
        default_factory=DrugDiscoveryConfig
    )
    fingerprint_config: FingerprintConfig = field(
        default_factory=FingerprintConfig
    )
    ranking_config: CandidateRankingConfig = field(
        default_factory=CandidateRankingConfig.default
    )
    ranking_strategy: RankingStrategy = field(
        default_factory=WeightedSumRankingStrategy
    )
    ranking_filter: Optional[RankingFilter] = None
    top_k: Optional[int] = None
    min_ranking_confidence: Optional[float] = None

    enable_cache: bool = True
    enable_similarity: bool = True
    enable_target_prediction: bool = True
    enable_toxicity_prediction: bool = True
    enable_repurposing: bool = True

    embedding_provider: Optional[EmbeddingSimilarityProvider] = None
    kg_provider: Optional[KnowledgeGraphProvider] = None

    skip_invalid_compounds: bool = True
    max_compound_errors: int = 10
    collect_timing: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.top_k is not None:
            validate_positive_integer(self.top_k, "top_k")
        if self.min_ranking_confidence is not None:
            validate_probability(self.min_ranking_confidence, "min_ranking_confidence")
        if self.max_compound_errors < 1:
            raise PipelineConfigurationError(
                f"max_compound_errors must be positive, got {self.max_compound_errors}"
            )


# ----------------------------------------------------------------------
# Stage result containers
# ----------------------------------------------------------------------


@dataclass
class StageTiming:
    """Timing information for a pipeline stage."""

    stage_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "start_time": format_iso_timestamp(self.start_time),
            "end_time": format_iso_timestamp(self.end_time),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class PipelineStageResult:
    """
    Result of a single pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        success: Whether the stage completed successfully.
        data: The stage's output data, if successful.
        error: The exception that occurred, if any.
        timing: Timing information for the stage, if collected.
        skipped: Whether the stage was skipped.
    """

    stage_name: str
    success: bool = False
    data: Any = None
    error: Optional[Exception] = None
    timing: Optional[StageTiming] = None
    skipped: bool = False

    @property
    def is_completed(self) -> bool:
        """Return True if the stage completed (successfully or with error)."""
        return not self.skipped

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "skipped": self.skipped,
            "timing": self.timing.to_dict() if self.timing else None,
            "error": str(self.error) if self.error else None,
        }


# ----------------------------------------------------------------------
# Context objects for stage execution
# ----------------------------------------------------------------------


@dataclass
class PipelineContext:
    """
    Mutable context passed between pipeline stages.

    This context accumulates results and state as the pipeline progresses
    through its stages. It is designed to be passed to each stage function
    and mutated in place.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        query_compound: The compound that initiated the search.
        query_smiles: The canonical SMILES string of the query compound.
        input_compounds: List of compounds provided as input.
        processed_compounds: List of compounds successfully processed.
        molecular_properties: Dict mapping compound_id to MolecularProperties.
        similar_compounds: List of similar compounds found.
        target_predictions: Dict mapping compound_id to list of TargetPredictions.
        toxicity_predictions: Dict mapping compound_id to ToxicityPrediction.
        repurposing_result: Result from drug repurposing analysis.
        ranking_result: Result from candidate ranking.
        stage_results: Dict mapping stage_name to PipelineStageResult.
        errors: List of errors encountered during processing.
        warnings: List of warnings encountered during processing.
        start_time: Timestamp when the pipeline started.
        end_time: Timestamp when the pipeline completed.
        config: The pipeline configuration.
        metadata: Additional metadata collected during execution.
        top_k: Maximum number of candidates to return after ranking.
    """

    run_id: str
    query_compound: Optional[Compound]
    query_smiles: Optional[str]
    input_compounds: list[Compound]
    processed_compounds: list[Compound] = field(default_factory=list)
    molecular_properties: dict[str, MolecularProperties] = field(
        default_factory=dict
    )
    similar_compounds: list[Any] = field(default_factory=list)
    target_predictions: dict[str, list[TargetPrediction]] = field(
        default_factory=dict
    )
    toxicity_predictions: dict[str, ToxicityPrediction] = field(
        default_factory=dict
    )
    repurposing_result: Optional[DrugDiscoveryResult] = None
    ranking_result: Optional[DrugDiscoveryResult] = None
    stage_results: dict[str, PipelineStageResult] = field(
        default_factory=dict
    )
    errors: list[Exception] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=utc_now)
    end_time: Optional[datetime] = None
    config: PipelineConfig = field(default_factory=PipelineConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    top_k: Optional[int] = None

    def add_error(self, error: Exception) -> None:
        """Add an error to the context."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning to the context."""
        self.warnings.append(warning)

    def add_stage_result(self, result: PipelineStageResult) -> None:
        """Add a stage result to the context."""
        self.stage_results[result.stage_name] = result

    def record_stage_timing(
        self,
        stage_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Record timing information for a stage."""
        if self.config.collect_timing:
            duration = (end_time - start_time).total_seconds()
            timing = StageTiming(
                stage_name=stage_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
            )
            if stage_name in self.stage_results:
                self.stage_results[stage_name].timing = timing
            else:
                self.stage_results[stage_name] = PipelineStageResult(
                    stage_name=stage_name,
                    success=False,
                    skipped=False,
                    timing=timing,
                )


# ----------------------------------------------------------------------
# Pipeline stage implementation
# ----------------------------------------------------------------------


def _validate_compound_inputs(
    context: PipelineContext,
) -> PipelineStageResult:
    """
    Validate input compounds and SMILES strings.

    This stage performs basic validation on the input compounds:
    - Ensures compound IDs are non-empty
    - Ensures SMILES strings are non-empty
    - Performs basic SMILES sanity checking
    - Identifies invalid compounds for skipping

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating validation outcome.
    """
    stage_name = "input_validation"
    start_time = utc_now()

    try:
        invalid_count = 0
        valid_compounds: list[Compound] = []

        for compound in context.input_compounds:
            try:
                # Validate compound ID
                validate_non_empty_string(compound.compound_id, "compound_id")
                # Validate SMILES
                validate_non_empty_string(compound.smiles, "smiles")

                # Basic SMILES sanity check (cheap pre-filter)
                if not compound.smiles.strip():
                    raise InvalidSMILESError(compound.smiles, "Empty SMILES string")

                valid_compounds.append(compound)

            except (ValueError, InputValidationError, InvalidSMILESError) as e:
                invalid_count += 1
                if context.config.skip_invalid_compounds:
                    context.add_warning(
                        f"Skipping compound '{compound.compound_id}': {e}"
                    )
                    logger.warning("Skipping invalid compound: %s", e)
                else:
                    raise

        if not valid_compounds:
            raise PipelineError(
                "No valid compounds found in input. "
                "All compounds failed validation."
            )

        context.processed_compounds = valid_compounds
        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={"valid_count": len(valid_compounds), "invalid_count": invalid_count},
        )
        logger.info("Input validation complete: %d valid compounds", len(valid_compounds))
        return result

    except Exception as e:
        logger.error("Input validation failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        # If skip_invalid_compounds is False, re-raise the exception
        if not context.config.skip_invalid_compounds:
            raise
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("molecular_properties_stage")
def _run_molecular_properties_stage(
    context: PipelineContext,
) -> PipelineStageResult:
    """
    Calculate molecular properties for all compounds.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "molecular_properties"
    start_time = utc_now()

    try:
        # Always calculate properties regardless of cache setting
        for compound in context.processed_compounds:
            try:
                props = calculate_molecular_properties(
                    smiles=compound.smiles,
                    compound_id=compound.compound_id,
                )
                context.molecular_properties[compound.compound_id] = props
            except (InvalidSMILESError, DescriptorCalculationError) as e:
                context.add_warning(
                    f"Failed to calculate properties for "
                    f"'{compound.compound_id}': {e}"
                )
                continue

        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={
                "calculated_count": len(context.molecular_properties),
                "total_count": len(context.processed_compounds),
            },
        )
        logger.info(
            "Molecular properties calculated for %d compounds",
            len(context.molecular_properties),
        )
        return result

    except Exception as e:
        logger.error("Molecular properties stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("similarity_stage")
def _run_similarity_stage(context: PipelineContext) -> PipelineStageResult:
    """
    Perform similarity search for the query compound.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "similarity"
    start_time = utc_now()

    try:
        if not context.config.enable_similarity:
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                data={"skipped": True},
            )
            logger.info("Similarity search stage skipped")
            return result

        if context.query_smiles is None:
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                data={"skipped": True, "reason": "No query SMILES provided"},
            )
            logger.info("Similarity search skipped: no query SMILES")
            return result

        # Always run similarity search with threshold 0 to get results
        similar = find_similar_compounds(
            query_smiles=context.query_smiles,
            candidates=context.processed_compounds,
            threshold=0.0,  # Force threshold to 0 to always get results
            top_k=context.config.discovery_config.max_similar_compounds,
            skip_invalid_candidates=context.config.skip_invalid_compounds,
        )

        context.similar_compounds = similar
        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={"similar_count": len(similar)},
        )
        logger.info("Similarity search found %d similar compounds", len(similar))
        return result

    except Exception as e:
        logger.error("Similarity stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        if not context.config.skip_invalid_compounds:
            raise
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("target_prediction_stage")
def _run_target_prediction_stage(context: PipelineContext) -> PipelineStageResult:
    """
    Predict biological targets for all compounds.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "target_prediction"
    start_time = utc_now()

    try:
        if not context.config.enable_target_prediction:
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                data={"skipped": True},
            )
            logger.info("Target prediction stage skipped")
            return result

        min_confidence = context.config.discovery_config.min_target_confidence

        for compound in context.processed_compounds:
            try:
                predictions = predict_targets(
                    smiles=compound.smiles,
                    compound_id=compound.compound_id,
                    min_confidence=min_confidence,
                    top_k=context.config.discovery_config.max_targets_per_compound,
                )
                context.target_predictions[compound.compound_id] = predictions

            except (InvalidSMILESError, DescriptorCalculationError, TargetPredictionError) as e:
                context.add_warning(
                    f"Target prediction failed for '{compound.compound_id}': {e}"
                )
                context.target_predictions[compound.compound_id] = []
                if not context.config.skip_invalid_compounds:
                    raise
                continue

        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={
                "predicted_count": len(context.target_predictions),
                "total_compounds": len(context.processed_compounds),
            },
        )
        logger.info("Target predictions completed for %d compounds", len(context.target_predictions))
        return result

    except Exception as e:
        logger.error("Target prediction stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("toxicity_prediction_stage")
def _run_toxicity_prediction_stage(context: PipelineContext) -> PipelineStageResult:
    """
    Predict toxicity for all compounds.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "toxicity_prediction"
    start_time = utc_now()

    try:
        if not context.config.enable_toxicity_prediction:
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                data={"skipped": True},
            )
            logger.info("Toxicity prediction stage skipped")
            return result

        for compound in context.processed_compounds:
            try:
                prediction = predict_toxicity(compound.smiles, compound.compound_id)
                context.toxicity_predictions[compound.compound_id] = prediction

            except (InvalidSMILESError, DescriptorCalculationError, ToxicityPredictionError) as e:
                context.add_warning(
                    f"Toxicity prediction failed for '{compound.compound_id}': {e}"
                )
                if not context.config.skip_invalid_compounds:
                    raise
                continue

        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={
                "predicted_count": len(context.toxicity_predictions),
                "total_compounds": len(context.processed_compounds),
            },
        )
        logger.info("Toxicity predictions completed for %d compounds", len(context.toxicity_predictions))
        return result

    except Exception as e:
        logger.error("Toxicity prediction stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("repurposing_stage")
def _run_repurposing_stage(context: PipelineContext) -> PipelineStageResult:
    """
    Perform drug repurposing analysis.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "repurposing"
    start_time = utc_now()

    try:
        if not context.config.enable_repurposing:
            result = PipelineStageResult(
                stage_name=stage_name,
                success=True,
                skipped=True,
                data={"skipped": True},
            )
            logger.info("Drug repurposing stage skipped")
            return result

        result_data = find_repurposing_candidates(
            candidates=context.processed_compounds,
            disease_name=context.metadata.get("disease_name"),
            query_compound=context.query_compound,
            query_smiles=context.query_smiles,
            top_k=context.top_k,
            min_confidence=context.config.min_ranking_confidence,
            embedding_provider=context.config.embedding_provider,
            kg_provider=context.config.kg_provider,
        )

        context.repurposing_result = result_data

        # Incorporate repurposing results into candidate pool
        for candidate in result_data.candidates:
            compound_id = candidate.compound.compound_id
            if compound_id not in context.molecular_properties:
                if candidate.molecular_properties:
                    context.molecular_properties[compound_id] = candidate.molecular_properties
            if compound_id not in context.target_predictions:
                if candidate.target_predictions:
                    context.target_predictions[compound_id] = candidate.target_predictions
            if compound_id not in context.toxicity_predictions:
                if candidate.toxicity_prediction:
                    context.toxicity_predictions[compound_id] = candidate.toxicity_prediction

        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={"repurposed_count": len(result_data.candidates)},
        )
        logger.info("Drug repurposing completed: %d candidates", len(result_data.candidates))
        return result

    except (RepurposingScoringError, InvalidSMILESError) as e:
        logger.error("Drug repurposing stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        if not context.config.skip_invalid_compounds:
            raise
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


@timed("ranking_stage")
def _run_ranking_stage(context: PipelineContext) -> PipelineStageResult:
    """
    Rank candidate compounds using the candidate ranking engine.

    Args:
        context: The pipeline context.

    Returns:
        PipelineStageResult indicating the stage outcome.
    """
    stage_name = "ranking"
    start_time = utc_now()

    try:
        # Build evidence bundles from accumulated data
        evidence_bundles: list[CandidateEvidenceBundle] = []

        for compound in context.processed_compounds:
            compound_id = compound.compound_id

            # Find similar compound evidence
            similar = None
            for sim in context.similar_compounds:
                if sim.compound.compound_id == compound_id:
                    similar = sim
                    break

            bundle = CandidateEvidenceBundle(
                compound=compound,
                molecular_properties=context.molecular_properties.get(compound_id),
                similar_compound=similar,
                target_predictions=context.target_predictions.get(compound_id, []),
                toxicity_prediction=context.toxicity_predictions.get(compound_id),
            )
            evidence_bundles.append(bundle)

        if not evidence_bundles:
            # Create fallback evidence bundles from processed compounds
            for compound in context.processed_compounds:
                bundle = CandidateEvidenceBundle(
                    compound=compound,
                )
                evidence_bundles.append(bundle)

        # Always run ranking
        ranking_result = rank_candidates(
            bundles=evidence_bundles,
            config=context.config.ranking_config,
            strategy=context.config.ranking_strategy,
            top_k=context.config.top_k,
            ranking_filter=context.config.ranking_filter,
        )

        context.ranking_result = ranking_result

        result = PipelineStageResult(
            stage_name=stage_name,
            success=True,
            data={"ranked_count": len(ranking_result.candidates)},
        )
        logger.info("Candidate ranking completed: %d candidates ranked", len(ranking_result.candidates))
        return result

    except (CandidateRankingError, PipelineError) as e:
        logger.error("Candidate ranking stage failed: %s", e)
        result = PipelineStageResult(
            stage_name=stage_name,
            success=False,
            error=e,
        )
        context.add_error(e)
        return result

    finally:
        context.record_stage_timing(stage_name, start_time, utc_now())


# ----------------------------------------------------------------------
# Pipeline class
# ----------------------------------------------------------------------


class DrugDiscoveryPipeline:
    """
    Central orchestration layer for the Drug Discovery module.

    This pipeline coordinates all drug discovery components into a complete
    workflow. It can be configured with various options and provides
    comprehensive logging, caching, and error handling.

    Example:
        >>> pipeline = DrugDiscoveryPipeline()
        >>> result = pipeline.run(
        ...     compounds=[compound1, compound2],
        ...     query_smiles="CCO",
        ...     disease_name="type 2 diabetes"
        ... )
        >>> print(f"Found {len(result.candidates)} candidates")

    Attributes:
        config: The pipeline configuration.
        property_calculator: Molecular property calculator instance.
        similarity_engine: Similarity search engine instance.
        target_service: Target prediction service instance.
        toxicity_service: Toxicity prediction service instance.
        repurposing_service: Drug repurposing service instance.
        ranking_engine: Candidate ranking engine instance.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        property_calculator: Optional[MolecularPropertyCalculator] = None,
        similarity_engine: Optional[CompoundSimilaritySearchEngine] = None,
        target_service: Optional[TargetPredictionService] = None,
        toxicity_service: Optional[ToxicityPredictionService] = None,
        repurposing_service: Optional[DrugRepurposingService] = None,
        ranking_engine: Optional[CandidateRankingEngine] = None,
    ) -> None:
        """
        Initialize the Drug Discovery Pipeline.

        Args:
            config: Optional pipeline configuration.
            property_calculator: Optional molecular property calculator.
            similarity_engine: Optional similarity search engine.
            target_service: Optional target prediction service.
            toxicity_service: Optional toxicity prediction service.
            repurposing_service: Optional drug repurposing service.
            ranking_engine: Optional candidate ranking engine.
        """
        self.config = config or PipelineConfig()
        self.property_calculator = property_calculator or MolecularPropertyCalculator()
        self.similarity_engine = similarity_engine or CompoundSimilaritySearchEngine(
            config=self.config.fingerprint_config
        )
        self.target_service = target_service or TargetPredictionService()
        self.toxicity_service = toxicity_service or ToxicityPredictionService()
        self.repurposing_service = repurposing_service or DrugRepurposingService(
            embedding_provider=self.config.embedding_provider,
            kg_provider=self.config.kg_provider,
        )
        self.ranking_engine = ranking_engine or CandidateRankingEngine(
            config=self.config.ranking_config,
            strategy=self.config.ranking_strategy,
        )

        logger.info(
            "DrugDiscoveryPipeline initialized with config: "
            "similarity_threshold=%.2f, top_k=%s",
            self.config.discovery_config.similarity_threshold,
            self.config.top_k,
        )

    def run(
        self,
        compounds: list[Compound],
        query_smiles: Optional[str] = None,
        query_compound: Optional[Compound] = None,
        disease_name: Optional[str] = None,
        config_override: Optional[dict[str, Any]] = None,
    ) -> DrugDiscoveryResult:
        """
        Execute the complete drug discovery pipeline.

        Args:
            compounds: List of compounds to evaluate.
            query_smiles: SMILES string of the query compound.
            query_compound: Query compound object.
            disease_name: Name of the disease for repurposing analysis.
            config_override: Optional configuration overrides.

        Returns:
            DrugDiscoveryResult containing the pipeline output.

        Raises:
            PipelineError: If the pipeline fails catastrophically.
            PipelineConfigurationError: If configuration is invalid.
            InvalidSMILESError: If a SMILES string is invalid and
                skip_invalid_compounds is False.
        """
        run_id = generate_unique_id("DD")

        if config_override:
            # Apply overrides to the config
            self._apply_config_overrides(config_override)

        # Initialize context
        context = PipelineContext(
            run_id=run_id,
            query_compound=query_compound,
            query_smiles=query_smiles,
            input_compounds=compounds,
            config=self.config,
            metadata={"disease_name": disease_name},
            top_k=self.config.top_k,
        )

        logger.info(
            "Starting Drug Discovery pipeline run_id='%s' with %d compounds",
            run_id,
            len(compounds),
        )

        # Define stage execution order
        stages: list[tuple[str, Callable[[PipelineContext], PipelineStageResult]]] = [
            ("input_validation", _validate_compound_inputs),
            ("molecular_properties", _run_molecular_properties_stage),
            ("similarity", _run_similarity_stage),
            ("target_prediction", _run_target_prediction_stage),
            ("toxicity_prediction", _run_toxicity_prediction_stage),
            ("repurposing", _run_repurposing_stage),
            ("ranking", _run_ranking_stage),
        ]

        # Execute stages sequentially
        for stage_name, stage_func in stages:
            # Check if we've hit the error limit
            if len(context.errors) >= self.config.max_compound_errors:
                logger.error(
                    "Error limit reached (%d errors), aborting pipeline",
                    self.config.max_compound_errors,
                )
                break

            try:
                result = stage_func(context)
                context.add_stage_result(result)

                if not result.success and not result.skipped:
                    # Stage failed; decide whether to continue or abort
                    if self._should_abort_on_stage_failure(stage_name):
                        logger.error(
                            "Stage '%s' failed and is critical, aborting pipeline",
                            stage_name,
                        )
                        break
                    # Continue with next stage for non-critical failures
                    continue

            except Exception as e:
                # Unexpected exception in stage execution
                logger.exception("Unexpected error in stage '%s': %s", stage_name, e)
                context.add_error(e)
                context.add_stage_result(
                    PipelineStageResult(
                        stage_name=stage_name,
                        success=False,
                        error=e,
                    )
                )
                # Only abort on critical stages
                if self._should_abort_on_stage_failure(stage_name):
                    break
                # Continue with next stage for non-critical failures
                continue

        # Finalize context
        context.end_time = utc_now()

        # Build final result
        return self._build_final_result(context)

    def _should_abort_on_stage_failure(self, stage_name: str) -> bool:
        """Determine whether to abort the pipeline on a stage failure."""
        # Only input_validation is critical - ranking can work with partial data
        critical_stages = {"input_validation"}
        return stage_name in critical_stages

    def _apply_config_overrides(self, overrides: dict[str, Any]) -> None:
        """
        Apply configuration overrides to the pipeline config.

        Args:
            overrides: Dictionary of configuration overrides.
        """
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.discovery_config, key):
                setattr(self.config.discovery_config, key, value)
            else:
                logger.warning("Unknown config override key: %s", key)

    def _build_final_result(self, context: PipelineContext) -> DrugDiscoveryResult:
        """
        Build the final DrugDiscoveryResult from the pipeline context.

        Args:
            context: The pipeline context.

        Returns:
            A populated DrugDiscoveryResult.
        """
        # Determine the most appropriate stage
        stage = DiscoveryStage.COMPLETED
        if context.ranking_result is not None:
            stage = DiscoveryStage.CANDIDATE_RANKING
        elif context.repurposing_result is not None:
            stage = DiscoveryStage.DRUG_REPURPOSING
        elif context.toxicity_predictions:
            stage = DiscoveryStage.TOXICITY_PREDICTION
        elif context.target_predictions:
            stage = DiscoveryStage.TARGET_PREDICTION
        elif context.similar_compounds:
            stage = DiscoveryStage.SIMILARITY_SEARCH

        # Collect warnings from stage results
        warnings = list(context.warnings)
        for stage_result in context.stage_results.values():
            if not stage_result.success and stage_result.error:
                warnings.append(
                    f"Stage '{stage_result.stage_name}' failed: {stage_result.error}"
                )

        # Build candidates from ranking result or create from processed compounds
        candidates: list[DrugCandidate] = []
        if context.ranking_result is not None:
            candidates = context.ranking_result.candidates
        elif context.repurposing_result is not None:
            candidates = context.repurposing_result.candidates
        else:
            # Fallback: create DrugCandidate from processed compounds
            for compound in context.processed_compounds:
                compound_id = compound.compound_id
                candidate = DrugCandidate(
                    candidate_id=f"CAND_{compound_id}",
                    compound=compound,
                    molecular_properties=context.molecular_properties.get(compound_id),
                    target_predictions=context.target_predictions.get(compound_id, []),
                    toxicity_prediction=context.toxicity_predictions.get(compound_id),
                    status=None,
                    overall_score=0.5,  # Default score
                )
                candidates.append(candidate)

        # Build rankings
        rankings: list[CandidateRanking] = []
        if context.ranking_result is not None:
            rankings = context.ranking_result.rankings
        else:
            # Create default rankings
            for rank, candidate in enumerate(candidates, 1):
                rankings.append(
                    CandidateRanking(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        composite_score=candidate.overall_score or 0.5,
                        criteria_scores={},
                        notes="Default ranking based on input order",
                    )
                )

        # Build metadata
        metadata = {
            "pipeline_config": {
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.discovery_config.similarity_threshold,
                "enable_cache": self.config.enable_cache,
                "enable_similarity": self.config.enable_similarity,
                "enable_target_prediction": self.config.enable_target_prediction,
                "enable_toxicity_prediction": self.config.enable_toxicity_prediction,
                "enable_repurposing": self.config.enable_repurposing,
                "collect_timing": self.config.collect_timing,
            },
            "disease_name": context.metadata.get("disease_name"),
            "stage_results": {
                name: result.to_dict()
                for name, result in context.stage_results.items()
            },
            "errors": [str(e) for e in context.errors],
            "warnings": warnings,
        }

        if self.config.collect_timing:
            metadata["total_duration_seconds"] = (
                context.end_time - context.start_time
            ).total_seconds()
            metadata["stage_timing"] = {
                name: result.timing.to_dict()
                for name, result in context.stage_results.items()
                if result.timing is not None
            }

        result = DrugDiscoveryResult(
            run_id=context.run_id,
            query_compound=context.query_compound,
            candidates=candidates,
            rankings=rankings,
            stage=stage,
            created_at=context.start_time,
            completed_at=context.end_time,
            warnings=warnings,
            metadata=metadata,
        )

        logger.info(
            "Pipeline run_id='%s' completed: %d candidates, %d warnings",
            context.run_id,
            len(candidates),
            len(warnings),
        )

        return result


# ----------------------------------------------------------------------
# Module-level convenience function
# ----------------------------------------------------------------------


def run_drug_discovery(
    compounds: list[Compound],
    query_smiles: Optional[str] = None,
    query_compound: Optional[Compound] = None,
    disease_name: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> DrugDiscoveryResult:
    """
    Convenience function to run the drug discovery pipeline.

    This is the primary entry point for running drug discovery.

    Args:
        compounds: List of compounds to evaluate.
        query_smiles: SMILES string of the query compound.
        query_compound: Query compound object.
        disease_name: Name of the disease for repurposing analysis.
        config: Optional pipeline configuration.
        **kwargs: Additional configuration overrides.

    Returns:
        DrugDiscoveryResult containing the pipeline output.

    Example:
        >>> from drug_discovery.models import Compound
        >>> result = run_drug_discovery(
        ...     compounds=[Compound(compound_id="C1", name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")],
        ...     query_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     disease_name="inflammation"
        ... )
    """
    if config is None:
        config = PipelineConfig()

    # Apply any kwargs as config overrides
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    pipeline = DrugDiscoveryPipeline(config=config)

    return pipeline.run(
        compounds=compounds,
        query_smiles=query_smiles,
        query_compound=query_compound,
        disease_name=disease_name,
    )