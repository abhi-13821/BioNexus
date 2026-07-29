"""
drug_discovery/toxicity_prediction.py

Rule-based toxicity risk assessment for the BioNexus Drug Discovery
module.

This module estimates the toxicity risk profile of a compound using
transparent, deterministic rules rather than a trained model. It is
built around the same pluggable-engine pattern used by
``drug_discovery.target_prediction`` so that a future machine learning
or deep learning toxicity model can be substituted for the rule-based
baseline without changing the public API relied upon by
``pipeline.py``, ``drug_info``, or ``smiles_analyzer``.

Responsibilities:
    - Validate and parse SMILES strings (reusing
      ``drug_discovery.molecular_properties`` for parsing and descriptor
      calculation, and ``drug_discovery.compound_similarity`` for
      fingerprint generation).
    - Evaluate Lipinski's Rule of Five compliance.
    - Detect PAINS (Pan-Assay Interference Compounds) and Brenk
      structural alerts using RDKit's built-in ``FilterCatalog``.
    - Estimate drug-likeness using the QED (Quantitative Estimate of
      Drug-likeness) score.
    - Combine these signals into a composite toxicity risk score and
      qualitative toxicity category.
    - Produce human-readable explanations supporting each prediction.
    - Return core results using the ``ToxicityPrediction`` dataclass
      defined in ``drug_discovery.models``.

Compatibility
-------------
Targets Python 3.11 and RDKit (rdkit-pypi / rdkit package, including the
``rdkit.Chem.FilterCatalog`` module).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from drug_discovery.compound_similarity import generate_morgan_fingerprint
from drug_discovery.models import PredictionConfidence, ToxicityLevel, ToxicityPrediction
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
    MolecularPropertyCalculator,
)
from drug_discovery.target_prediction import MolecularFeatures

__all__ = [
    "InvalidSMILESError",
    "DescriptorCalculationError",
    "ToxicityPredictionError",
    "StructuralAlert",
    "ToxicityAssessmentReport",
    "ToxicityPredictionEngine",
    "RuleBasedToxicityPredictor",
    "ToxicityPredictionService",
    "predict_toxicity",
    "predict_toxicity_batch",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ToxicityPredictionError(RuntimeError):
    """
    Raised when toxicity prediction fails unexpectedly for an otherwise
    valid, successfully parsed molecule.

    Attributes:
        compound_id: Identifier of the compound for which prediction
            failed.
        engine_name: Name of the prediction engine that raised the
            failure.
    """

    def __init__(
        self,
        compound_id: str,
        engine_name: str,
        message: str | None = None,
    ) -> None:
        self.compound_id = compound_id
        self.engine_name = engine_name
        resolved_message = (
            message
            or f"Toxicity prediction failed for compound_id='{compound_id}' "
            f"using engine '{engine_name}'."
        )
        super().__init__(resolved_message)


# ---------------------------------------------------------------------------
# Supporting data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralAlert:
    """
    A single structural alert flagged by a substructure filter catalog
    (e.g., PAINS or Brenk).

    Attributes:
        alert_name: Short identifier of the matched alert, as reported by
            the filter catalog (e.g., "quinone_A").
        catalog_name: Name of the catalog the alert originated from
            (e.g., "PAINS", "Brenk").
        description: Human-readable description of the structural
            concern represented by the alert.
    """

    alert_name: str
    catalog_name: str
    description: str


@dataclass
class ToxicityAssessmentReport:
    """
    A complete toxicity assessment for a single compound, combining the
    core ``ToxicityPrediction`` result (as defined in
    ``drug_discovery.models``) with supplementary, human-readable
    context.

    This richer report is what prediction engines return internally;
    ``ToxicityPredictionService`` exposes both this full report and a
    convenience accessor that returns only the ``ToxicityPrediction``
    for callers that only need the standard model schema.

    Attributes:
        compound_id: Identifier of the assessed compound.
        prediction: The core ``ToxicityPrediction`` result.
        lipinski_compliant: Whether the compound satisfies Lipinski's
            Rule of Five (zero violations).
        lipinski_violations: Number of Lipinski's Rule of Five
            violations.
        drug_likeness_score: QED (Quantitative Estimate of
            Drug-likeness) score, in [0.0, 1.0].
        structural_alerts: Structural alerts detected via PAINS/Brenk
            substructure filtering.
        explanations: Human-readable explanations supporting the
            prediction, describing which factors contributed to the
            assigned toxicity score and category.
    """

    compound_id: str
    prediction: ToxicityPrediction
    lipinski_compliant: bool
    lipinski_violations: int
    drug_likeness_score: float
    structural_alerts: list[StructuralAlert] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate supplementary report fields."""
        if not self.compound_id.strip():
            raise ValueError("compound_id must be a non-empty string.")
        if self.lipinski_violations < 0:
            raise ValueError("lipinski_violations cannot be negative.")
        if not 0.0 <= self.drug_likeness_score <= 1.0:
            raise ValueError("drug_likeness_score must be within [0.0, 1.0].")


# ---------------------------------------------------------------------------
# Prediction engine interface
# ---------------------------------------------------------------------------


class ToxicityPredictionEngine(ABC):
    """
    Abstract interface for compound toxicity prediction backends.

    Any prediction backend -- rule-based, classical machine learning, or
    deep learning -- must implement this interface so that it can be
    used interchangeably by ``ToxicityPredictionService`` and, by
    extension, ``pipeline.py``. A future ML-based engine may return an
    empty ``structural_alerts``/``explanations`` list if such detail is
    not naturally produced by the underlying model, while still
    populating the required ``ToxicityPrediction`` fields.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """
        Return a short, human-readable identifier for this engine.

        Returns:
            The engine's name (e.g., "rule_based_v1", "toxbert_v1").
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: MolecularFeatures) -> ToxicityAssessmentReport:
        """
        Predict the toxicity risk profile of a single compound.

        Args:
            features: The ``MolecularFeatures`` representation of the
                compound to evaluate.

        Returns:
            A populated ``ToxicityAssessmentReport``.

        Raises:
            ToxicityPredictionError: If prediction fails unexpectedly.
        """
        raise NotImplementedError


class RuleBasedToxicityPredictor(ToxicityPredictionEngine):
    """
    Baseline rule-based toxicity prediction engine.

    Combines four transparent, deterministic signals into a composite
    toxicity risk score:

        1. Lipinski's Rule of Five violations (drug-likeness proxy).
        2. PAINS/Brenk structural alerts (assay interference and
           reactive/toxic substructure liabilities), via RDKit's
           ``FilterCatalog``.
        3. Lipophilicity risk, derived from calculated LogP (high
           lipophilicity is associated with elevated off-target and
           hepatotoxicity risk).
        4. Drug-likeness risk, derived from the QED score (lower
           drug-likeness is treated as an elevated risk signal).

    This engine requires no trained model and is intended to keep the
    ``toxicity_prediction`` module fully functional out of the box. It
    is intentionally conservative: it is designed to be better at
    flagging potential concerns than at confidently certifying safety,
    which is reflected in how prediction confidence is assigned.

    Attributes:
        weights: Mapping of the four endpoint names to their contribution
            weight in the composite toxicity score. Weights sum to 1.0.
    """

    _ENGINE_NAME: str = "rule_based_v1"

    # Composite score weighting. Structural alerts and Lipinski violations
    # are weighted most heavily as they are the most established
    # medicinal-chemistry heuristics for liability screening.
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "structural_alert_risk": 0.35,
        "lipinski_risk": 0.20,
        "lipophilicity_risk": 0.20,
        "drug_likeness_risk": 0.25,
    }

    # LogP value at which lipophilicity risk begins to accrue, and the
    # value at which it saturates at maximum risk (1.0).
    _LOGP_RISK_ONSET: float = 3.0
    _LOGP_RISK_SATURATION: float = 10.0

    _RISK_PER_LIPINSKI_VIOLATION: float = 0.25
    _RISK_PER_STRUCTURAL_ALERT: float = 0.35

    _LOW_MODERATE_BOUNDARY: float = 0.30
    _MODERATE_HIGH_BOUNDARY: float = 0.60
    _HIGH_SEVERE_BOUNDARY: float = 0.85

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """
        Initialize the rule-based toxicity predictor.

        Args:
            weights: Optional custom weighting for the four risk
                endpoints ("structural_alert_risk", "lipinski_risk",
                "lipophilicity_risk", "drug_likeness_risk"). If omitted,
                the default weighting is used. Weights should sum to
                approximately 1.0.

        Raises:
            ValueError: If any provided weight is negative.
        """
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)
        for endpoint, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Weight for '{endpoint}' cannot be negative.")

        self._filter_catalog = self._build_filter_catalog()
        self._property_calculator = MolecularPropertyCalculator()

    @property
    def engine_name(self) -> str:
        """Return the engine's identifier."""
        return self._ENGINE_NAME

    @staticmethod
    def _build_filter_catalog() -> FilterCatalog:
        """
        Build the combined PAINS + Brenk RDKit structural alert filter
        catalog.

        Returns:
            A configured ``FilterCatalog`` instance covering both the
            PAINS and Brenk substructure filter sets.
        """
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        return FilterCatalog(params)

    def _detect_structural_alerts(
        self, features: MolecularFeatures
    ) -> list[StructuralAlert]:
        """
        Detect PAINS and Brenk structural alerts for a molecule.

        Args:
            features: The ``MolecularFeatures`` representation of the
                compound to evaluate.

        Returns:
            A list of ``StructuralAlert`` instances, one per matched
            filter catalog entry.

        Raises:
            ToxicityPredictionError: If substructure filtering fails
                unexpectedly.
        """
        try:
            matches = self._filter_catalog.GetMatches(features.mol)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Structural alert detection failed for compound_id='%s'.",
                features.compound_id,
            )
            raise ToxicityPredictionError(
                compound_id=features.compound_id,
                engine_name=self.engine_name,
                message="PAINS/Brenk structural alert detection failed.",
            ) from exc

        alerts: list[StructuralAlert] = []
        for match in matches:
            description = match.GetDescription()
            catalog_entries = match.GetPropList()
            catalog_name = "PAINS/Brenk"
            if "FilterSet" in catalog_entries:
                catalog_name = str(match.GetProp("FilterSet"))
            alerts.append(
                StructuralAlert(
                    alert_name=description,
                    catalog_name=catalog_name,
                    description=(
                        f"Structural pattern '{description}' matched a "
                        f"known assay-interference or reactivity liability "
                        f"from the {catalog_name} filter set."
                    ),
                )
            )
        return alerts

    @classmethod
    def _lipophilicity_risk(cls, logp: float) -> float:
        """
        Convert a calculated LogP value into a normalized lipophilicity
        risk score.

        Risk is zero at or below the onset threshold, rises linearly,
        and saturates at 1.0 at or above the saturation threshold.

        Args:
            logp: Calculated octanol-water partition coefficient.

        Returns:
            A lipophilicity risk score, in [0.0, 1.0].
        """
        if logp <= cls._LOGP_RISK_ONSET:
            return 0.0
        span = cls._LOGP_RISK_SATURATION - cls._LOGP_RISK_ONSET
        risk = (logp - cls._LOGP_RISK_ONSET) / span
        return min(1.0, max(0.0, risk))

    @classmethod
    def _score_to_level(cls, toxicity_score: float) -> ToxicityLevel:
        """
        Map a composite toxicity score to a qualitative toxicity
        category.

        The primary categories requested are Low, Moderate, and High.
        This mapping additionally uses the ``SEVERE`` level already
        defined in ``drug_discovery.models.ToxicityLevel`` for scores at
        the extreme upper end of the range, providing finer resolution
        for downstream ranking without contradicting the requested
        three-tier scheme (SEVERE compounds are still a subset of "high
        risk").

        Args:
            toxicity_score: Composite toxicity risk score, in
                [0.0, 1.0].

        Returns:
            The corresponding ``ToxicityLevel``.
        """
        if toxicity_score >= cls._HIGH_SEVERE_BOUNDARY:
            return ToxicityLevel.SEVERE
        if toxicity_score >= cls._MODERATE_HIGH_BOUNDARY:
            return ToxicityLevel.HIGH
        if toxicity_score >= cls._LOW_MODERATE_BOUNDARY:
            return ToxicityLevel.MODERATE
        return ToxicityLevel.LOW

    @staticmethod
    def _evidence_to_confidence(evidence_count: int) -> PredictionConfidence:
        """
        Map the amount of corroborating rule-based evidence to a
        qualitative confidence bucket.

        Rule-based heuristics are generally more reliable at flagging a
        concern than at certifying the absence of one, so confidence
        scales with the number of independent red flags observed
        (structural alerts plus Lipinski violations) rather than with
        the toxicity score itself.

        Args:
            evidence_count: Combined count of structural alerts and
                Lipinski violations observed for the compound.

        Returns:
            The corresponding ``PredictionConfidence`` bucket.
        """
        if evidence_count >= 5:
            return PredictionConfidence.VERY_HIGH
        if evidence_count >= 3:
            return PredictionConfidence.HIGH
        if evidence_count >= 1:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def predict(self, features: MolecularFeatures) -> ToxicityAssessmentReport:
        """
        Predict the toxicity risk profile of a compound using rule-based
        heuristics.

        Args:
            features: The ``MolecularFeatures`` representation of the
                compound to evaluate.

        Returns:
            A populated ``ToxicityAssessmentReport``.

        Raises:
            ToxicityPredictionError: If descriptor calculation or
                structural alert detection fails unexpectedly.
        """
        try:
            molecular_properties = self._property_calculator.calculate(
                smiles=features.smiles, compound_id=features.compound_id
            )
        except (InvalidSMILESError, DescriptorCalculationError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Descriptor calculation failed during toxicity prediction "
                "for compound_id='%s'.",
                features.compound_id,
            )
            raise ToxicityPredictionError(
                compound_id=features.compound_id, engine_name=self.engine_name
            ) from exc

        structural_alerts = self._detect_structural_alerts(features)
        alert_count = len(structural_alerts)
        lipinski_violations = molecular_properties.lipinski_violations
        drug_likeness_score = (
            molecular_properties.qed_score
            if molecular_properties.qed_score is not None
            else 0.5
        )

        endpoint_scores: dict[str, float] = {
            "structural_alert_risk": min(
                1.0, alert_count * self._RISK_PER_STRUCTURAL_ALERT
            ),
            "lipinski_risk": min(
                1.0, lipinski_violations * self._RISK_PER_LIPINSKI_VIOLATION
            ),
            "lipophilicity_risk": self._lipophilicity_risk(molecular_properties.logp),
            "drug_likeness_risk": 1.0 - drug_likeness_score,
        }

        toxicity_score = min(
            1.0,
            sum(
                endpoint_scores[endpoint] * weight
                for endpoint, weight in self.weights.items()
                if endpoint in endpoint_scores
            ),
        )
        toxicity_level = self._score_to_level(toxicity_score)

        evidence_count = alert_count + lipinski_violations
        confidence = self._evidence_to_confidence(evidence_count)

        explanations = self._build_explanations(
            molecular_properties_logp=molecular_properties.logp,
            molecular_properties_mw=molecular_properties.molecular_weight,
            lipinski_violations=lipinski_violations,
            structural_alerts=structural_alerts,
            drug_likeness_score=drug_likeness_score,
            toxicity_score=toxicity_score,
            toxicity_level=toxicity_level,
        )

        logger.info(
            "Toxicity assessment for compound_id='%s': score=%.3f, "
            "level=%s, alerts=%d, lipinski_violations=%d.",
            features.compound_id,
            toxicity_score,
            toxicity_level.value,
            alert_count,
            lipinski_violations,
        )

        prediction = ToxicityPrediction(
            compound_id=features.compound_id,
            toxicity_level=toxicity_level,
            toxicity_score=toxicity_score,
            endpoint_scores=endpoint_scores,
            confidence=confidence,
            prediction_method=self.engine_name,
        )

        return ToxicityAssessmentReport(
            compound_id=features.compound_id,
            prediction=prediction,
            lipinski_compliant=(lipinski_violations == 0),
            lipinski_violations=lipinski_violations,
            drug_likeness_score=drug_likeness_score,
            structural_alerts=structural_alerts,
            explanations=explanations,
        )

    @staticmethod
    def _build_explanations(
        molecular_properties_logp: float,
        molecular_properties_mw: float,
        lipinski_violations: int,
        structural_alerts: list[StructuralAlert],
        drug_likeness_score: float,
        toxicity_score: float,
        toxicity_level: ToxicityLevel,
    ) -> list[str]:
        """
        Build a list of human-readable explanations describing the
        factors that contributed to a toxicity assessment.

        Args:
            molecular_properties_logp: Calculated LogP value.
            molecular_properties_mw: Calculated molecular weight.
            lipinski_violations: Number of Lipinski's Rule of Five
                violations.
            structural_alerts: Structural alerts detected for the
                compound.
            drug_likeness_score: QED drug-likeness score.
            toxicity_score: Final composite toxicity risk score.
            toxicity_level: Final assigned toxicity category.

        Returns:
            A list of explanation strings, ordered from overall summary
            to specific contributing factors.
        """
        explanations: list[str] = [
            f"Overall toxicity risk score is {toxicity_score:.2f} "
            f"(category: {toxicity_level.value})."
        ]

        if lipinski_violations == 0:
            explanations.append(
                "Compound is fully compliant with Lipinski's Rule of Five."
            )
        else:
            explanations.append(
                f"Compound violates {lipinski_violations} of Lipinski's "
                f"Rule of Five criteria (MW={molecular_properties_mw:.1f}, "
                f"LogP={molecular_properties_logp:.2f}), which may indicate "
                f"reduced drug-likeness and elevated risk."
            )

        if structural_alerts:
            alert_names = ", ".join(alert.alert_name for alert in structural_alerts)
            explanations.append(
                f"{len(structural_alerts)} PAINS/Brenk structural alert(s) "
                f"detected: {alert_names}. These substructures are "
                f"associated with assay interference or known reactivity "
                f"and toxicity liabilities."
            )
        else:
            explanations.append(
                "No PAINS or Brenk structural alerts were detected."
            )

        if molecular_properties_logp > RuleBasedToxicityPredictor._LOGP_RISK_ONSET:
            explanations.append(
                f"Calculated LogP of {molecular_properties_logp:.2f} exceeds "
                f"{RuleBasedToxicityPredictor._LOGP_RISK_ONSET:.1f}, "
                f"indicating elevated lipophilicity that is commonly "
                f"associated with off-target binding and hepatotoxicity "
                f"risk."
            )

        if drug_likeness_score < 0.5:
            explanations.append(
                f"QED drug-likeness score of {drug_likeness_score:.2f} is "
                f"relatively low, suggesting the compound's overall "
                f"physicochemical profile is less favorable for further "
                f"development."
            )
        else:
            explanations.append(
                f"QED drug-likeness score of {drug_likeness_score:.2f} "
                f"indicates a reasonably favorable physicochemical profile."
            )

        return explanations


# ---------------------------------------------------------------------------
# Prediction service (orchestration layer)
# ---------------------------------------------------------------------------


class ToxicityPredictionService:
    """
    Orchestrates SMILES validation, feature extraction, and toxicity
    prediction using a pluggable ``ToxicityPredictionEngine``.

    This is the primary integration point for ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``. The prediction backend can
    be swapped (e.g., for a trained machine learning or deep learning
    model) by constructing this service with a different
    ``ToxicityPredictionEngine`` implementation, without changing any
    calling code.

    Attributes:
        engine: The active ``ToxicityPredictionEngine`` used for
            prediction.
        fingerprint_radius: Morgan fingerprint radius used for feature
            extraction.
        fingerprint_n_bits: Morgan fingerprint bit-vector length used for
            feature extraction.
    """

    def __init__(
        self,
        engine: ToxicityPredictionEngine | None = None,
        fingerprint_radius: int = 2,
        fingerprint_n_bits: int = 2048,
    ) -> None:
        """
        Initialize the toxicity prediction service.

        Args:
            engine: The prediction engine to use. If omitted, a
                ``RuleBasedToxicityPredictor`` baseline is used.
            fingerprint_radius: Morgan fingerprint radius used when
                extracting features (default 2, i.e. ECFP4).
            fingerprint_n_bits: Morgan fingerprint bit-vector length used
                when extracting features (default 2048).
        """
        self.engine: ToxicityPredictionEngine = engine or RuleBasedToxicityPredictor()
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_n_bits = fingerprint_n_bits
        self._property_calculator = MolecularPropertyCalculator()

    def extract_features(self, smiles: str, compound_id: str) -> MolecularFeatures:
        """
        Validate a SMILES string and extract the molecular feature
        representation used for prediction.

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound the features belong
                to.

        Returns:
            A populated ``MolecularFeatures`` instance.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly for a successfully parsed molecule.
        """
        mol = self._property_calculator.parse_smiles(smiles)
        fingerprint = generate_morgan_fingerprint(
            smiles, radius=self.fingerprint_radius, n_bits=self.fingerprint_n_bits
        )
        return MolecularFeatures(
            compound_id=compound_id,
            smiles=smiles.strip(),
            mol=mol,
            fingerprint=fingerprint,
        )

    def assess_toxicity(
        self, smiles: str, compound_id: str
    ) -> ToxicityAssessmentReport:
        """
        Produce a full toxicity assessment report for a single compound,
        including structural alerts and human-readable explanations.

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound being evaluated.

        Returns:
            A populated ``ToxicityAssessmentReport``.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If descriptor or fingerprint
                generation fails unexpectedly.
            ToxicityPredictionError: If the underlying engine fails
                unexpectedly during prediction.
        """
        logger.info(
            "Assessing toxicity for compound_id='%s' using engine '%s'.",
            compound_id,
            self.engine.engine_name,
        )

        features = self.extract_features(smiles, compound_id)

        try:
            report = self.engine.predict(features)
        except (InvalidSMILESError, DescriptorCalculationError, ToxicityPredictionError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Unexpected engine failure for compound_id='%s'.", compound_id
            )
            raise ToxicityPredictionError(
                compound_id=compound_id, engine_name=self.engine.engine_name
            ) from exc

        return report

    def predict_toxicity(self, smiles: str, compound_id: str) -> ToxicityPrediction:
        """
        Predict the toxicity risk profile of a single compound, returning
        only the core ``ToxicityPrediction`` result defined in
        ``drug_discovery.models``.

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound being evaluated.

        Returns:
            A populated ``ToxicityPrediction`` instance.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If descriptor or fingerprint
                generation fails unexpectedly.
            ToxicityPredictionError: If the underlying engine fails
                unexpectedly during prediction.
        """
        return self.assess_toxicity(smiles=smiles, compound_id=compound_id).prediction

    def predict_toxicity_batch(
        self,
        compounds: list[tuple[str, str]],
        skip_invalid: bool = True,
    ) -> dict[str, ToxicityPrediction]:
        """
        Predict the toxicity risk profile for multiple compounds.

        Args:
            compounds: A list of ``(smiles, compound_id)`` tuples to
                evaluate.
            skip_invalid: If True, compounds with unparseable SMILES are
                logged and skipped rather than raising an exception. If
                False, an invalid SMILES causes ``InvalidSMILESError`` to
                propagate.

        Returns:
            A mapping of ``compound_id`` to its ``ToxicityPrediction``.

        Raises:
            InvalidSMILESError: If a compound's SMILES cannot be parsed
                and ``skip_invalid`` is False.
            DescriptorCalculationError: If descriptor or fingerprint
                generation fails unexpectedly for a compound.
            ToxicityPredictionError: If the underlying engine fails
                unexpectedly during prediction for a compound.
        """
        results: dict[str, ToxicityPrediction] = {}

        for smiles, compound_id in compounds:
            try:
                results[compound_id] = self.predict_toxicity(
                    smiles=smiles, compound_id=compound_id
                )
            except InvalidSMILESError:
                if skip_invalid:
                    logger.warning(
                        "Skipping compound_id='%s' with invalid SMILES: '%s'",
                        compound_id,
                        smiles,
                    )
                    continue
                raise

        return results


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_service = ToxicityPredictionService()


def predict_toxicity(smiles: str, compound_id: str) -> ToxicityPrediction:
    """
    Predict the toxicity risk profile of a single compound using the
    default rule-based prediction service.

    This is the primary entry point intended for use by ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``.

    Args:
        smiles: The SMILES string representing the compound.
        compound_id: Identifier of the compound being evaluated.

    Returns:
        A populated ``ToxicityPrediction`` instance.

    Raises:
        InvalidSMILESError: If the SMILES string cannot be parsed.
        DescriptorCalculationError: If descriptor or fingerprint
            generation fails unexpectedly.
        ToxicityPredictionError: If the underlying engine fails
            unexpectedly during prediction.
    """
    return _default_service.predict_toxicity(smiles=smiles, compound_id=compound_id)


def predict_toxicity_batch(
    compounds: list[tuple[str, str]],
    skip_invalid: bool = True,
) -> dict[str, ToxicityPrediction]:
    """
    Predict the toxicity risk profile for multiple compounds using the
    default rule-based prediction service.

    Args:
        compounds: A list of ``(smiles, compound_id)`` tuples to
            evaluate.
        skip_invalid: If True, compounds with unparseable SMILES are
            logged and skipped rather than raising an exception.

    Returns:
        A mapping of ``compound_id`` to its ``ToxicityPrediction``.

    Raises:
        InvalidSMILESError: If a compound's SMILES cannot be parsed and
            ``skip_invalid`` is False.
        DescriptorCalculationError: If descriptor or fingerprint
            generation fails unexpectedly for a compound.
        ToxicityPredictionError: If the underlying engine fails
            unexpectedly during prediction for a compound.
    """
    return _default_service.predict_toxicity_batch(
        compounds=compounds, skip_invalid=skip_invalid
    )