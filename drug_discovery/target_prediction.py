"""
drug_discovery/target_prediction.py

Compound-target interaction prediction for the BioNexus Drug Discovery
module.

This module predicts which biological targets (proteins, enzymes,
receptors, etc.) a given compound may interact with. It is designed
around a pluggable prediction-engine interface so that the initial
rule-based baseline predictor can later be swapped for, or supplemented
by, a trained machine learning or deep learning model without changing
any calling code in ``pipeline.py``, ``drug_info``, or
``smiles_analyzer``.

Responsibilities:
    - Validate and parse SMILES strings (reusing
      ``drug_discovery.molecular_properties`` for parsing and
      ``drug_discovery.compound_similarity`` for fingerprint generation).
    - Extract a feature representation (RDKit molecule + Morgan
      fingerprint) suitable for both rule-based and future ML-based
      engines.
    - Define a ``TargetPredictionEngine`` abstract interface that any
      prediction backend must implement.
    - Provide a ``RuleBasedTargetPredictor`` baseline implementation that
      uses SMARTS substructure matching against a curated knowledge base
      of structural motifs associated with known target classes.
    - Orchestrate prediction for single compounds and batches, returning
      ranked results as ``TargetPrediction`` instances.

Compatibility
-------------
Targets Python 3.11 and RDKit (rdkit-pypi / rdkit package).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from drug_discovery.compound_similarity import generate_morgan_fingerprint
from drug_discovery.models import (
    DrugTarget,
    PredictionConfidence,
    TargetPrediction,
    TargetType,
)
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
    MolecularPropertyCalculator,
)

__all__ = [
    "InvalidSMILESError",
    "DescriptorCalculationError",
    "TargetPredictionError",
    "MolecularFeatures",
    "TargetPredictionRule",
    "TargetPredictionEngine",
    "RuleBasedTargetPredictor",
    "TargetPredictionService",
    "predict_targets",
    "predict_targets_batch",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TargetPredictionError(RuntimeError):
    """
    Raised when target prediction fails unexpectedly for an otherwise
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
            or f"Target prediction failed for compound_id='{compound_id}' "
            f"using engine '{engine_name}'."
        )
        super().__init__(resolved_message)


# ---------------------------------------------------------------------------
# Feature representation
# ---------------------------------------------------------------------------


@dataclass
class MolecularFeatures:
    """
    Feature representation of a compound used as input to target
    prediction engines.

    This representation is deliberately engine-agnostic: rule-based
    engines can operate on ``mol`` directly via substructure matching,
    while future machine learning or deep learning engines can consume
    ``fingerprint`` (or extend this class with additional feature
    tensors).

    Attributes:
        compound_id: Identifier of the compound the features belong to.
        smiles: The (cleaned) SMILES string the features were derived
            from.
        mol: The parsed RDKit molecule object.
        fingerprint: A Morgan (ECFP4-equivalent) fingerprint bit vector.
    """

    compound_id: str
    smiles: str
    mol: Chem.Mol
    fingerprint: ExplicitBitVect


# ---------------------------------------------------------------------------
# Rule-based knowledge base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetPredictionRule:
    """
    A single structural rule used by ``RuleBasedTargetPredictor``,
    associating a SMARTS substructure pattern with a candidate biological
    target.

    Attributes:
        rule_id: Unique identifier for this rule.
        smarts_pattern: SMARTS pattern describing the structural motif
            associated with the target.
        target: The ``DrugTarget`` predicted when the pattern matches.
        base_confidence: Baseline confidence score, in [0.0, 1.0],
            assigned when the pattern matches exactly once.
        description: Human-readable rationale for the rule.
    """

    rule_id: str
    smarts_pattern: str
    target: DrugTarget
    base_confidence: float
    description: str

    def __post_init__(self) -> None:
        """Validate the rule's structural pattern and confidence value."""
        if not self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string.")
        if Chem.MolFromSmarts(self.smarts_pattern) is None:
            raise ValueError(
                f"Invalid SMARTS pattern for rule '{self.rule_id}': "
                f"'{self.smarts_pattern}'"
            )
        if not 0.0 <= self.base_confidence <= 1.0:
            raise ValueError("base_confidence must be within [0.0, 1.0].")


def _build_default_rule_knowledge_base() -> list[TargetPredictionRule]:
    """
    Construct the default curated knowledge base of structural motifs
    mapped to candidate target classes, used by
    ``RuleBasedTargetPredictor``.

    This baseline knowledge base is intentionally simple and
    illustrative; it is intended to make the module functional without
    requiring a trained model, and is designed to be replaced or
    augmented by richer rule sets or ML-derived rules over time.

    Returns:
        A list of ``TargetPredictionRule`` instances.
    """
    return [
        TargetPredictionRule(
            rule_id="beta_lactam_ring",
            smarts_pattern="[#6]1[#6][#6](=O)[#7]1",
            target=DrugTarget(
                target_id="TGT_PBP",
                name="Penicillin-binding protein",
                target_type=TargetType.ENZYME,
                organism="Bacteria",
                description=(
                    "Transpeptidase enzymes involved in bacterial cell "
                    "wall synthesis, classically inhibited by beta-lactam "
                    "antibiotics."
                ),
            ),
            base_confidence=0.75,
            description="Beta-lactam (4-membered cyclic amide) ring detected.",
        ),
        TargetPredictionRule(
            rule_id="sulfonamide_group",
            smarts_pattern="[#16](=O)(=O)[#7]",
            target=DrugTarget(
                target_id="TGT_CA",
                name="Carbonic anhydrase",
                target_type=TargetType.ENZYME,
                organism="Homo sapiens",
                gene_symbol="CA2",
                description=(
                    "Zinc metalloenzyme frequently inhibited by "
                    "sulfonamide-containing compounds."
                ),
            ),
            base_confidence=0.6,
            description="Sulfonamide (S(=O)(=O)N) functional group detected.",
        ),
        TargetPredictionRule(
            rule_id="aminopyrimidine_hinge_binder",
            smarts_pattern="c1ncncc1[#7]",
            target=DrugTarget(
                target_id="TGT_KINASE",
                name="Protein kinase (ATP-binding site)",
                target_type=TargetType.ENZYME,
                organism="Homo sapiens",
                description=(
                    "Aminopyrimidine motifs are common hinge-binding "
                    "fragments in ATP-competitive kinase inhibitors."
                ),
            ),
            base_confidence=0.55,
            description="Aminopyrimidine kinase-hinge-binding motif detected.",
        ),
        TargetPredictionRule(
            rule_id="carboxylic_acid_aromatic",
            smarts_pattern="c[CX3](=O)[OX2H1]",
            target=DrugTarget(
                target_id="TGT_COX",
                name="Cyclooxygenase (COX-1/COX-2)",
                target_type=TargetType.ENZYME,
                organism="Homo sapiens",
                gene_symbol="PTGS2",
                description=(
                    "Aromatic carboxylic acids are a recurring "
                    "pharmacophore element in NSAID COX inhibitors."
                ),
            ),
            base_confidence=0.5,
            description="Aromatic carboxylic acid group detected.",
        ),
        TargetPredictionRule(
            rule_id="benzodiazepine_core",
            smarts_pattern="c1ccc2c(c1)C(=N/CC(=O)N2)",
            target=DrugTarget(
                target_id="TGT_GABAA",
                name="GABA-A receptor",
                target_type=TargetType.RECEPTOR,
                organism="Homo sapiens",
                description=(
                    "The fused benzodiazepine ring system is characteristic "
                    "of GABA-A receptor allosteric modulators."
                ),
            ),
            base_confidence=0.65,
            description="Benzodiazepine fused ring core detected.",
        ),
        TargetPredictionRule(
            rule_id="steroid_nucleus",
            smarts_pattern=(
                "C1CCC2C(C1)CCC3C2CCC4C3(CCCC4)"
            ),
            target=DrugTarget(
                target_id="TGT_STEROID_RECEPTOR",
                name="Nuclear steroid hormone receptor",
                target_type=TargetType.RECEPTOR,
                organism="Homo sapiens",
                description=(
                    "The cyclopentanoperhydrophenanthrene (steroid) "
                    "nucleus is characteristic of steroid hormone "
                    "receptor ligands."
                ),
            ),
            base_confidence=0.6,
            description="Steroid nucleus (four fused rings) detected.",
        ),
        TargetPredictionRule(
            rule_id="sulfonyl_urea",
            smarts_pattern="[#16](=O)(=O)[#7][#6](=O)[#7]",
            target=DrugTarget(
                target_id="TGT_SUR1",
                name="Sulfonylurea receptor (ABC transporter subunit)",
                target_type=TargetType.TRANSPORTER,
                organism="Homo sapiens",
                gene_symbol="ABCC8",
                description=(
                    "Sulfonylurea moieties are the defining pharmacophore "
                    "of antidiabetic agents targeting the sulfonylurea "
                    "receptor."
                ),
            ),
            base_confidence=0.7,
            description="Sulfonylurea functional group detected.",
        ),
        TargetPredictionRule(
            rule_id="phosphonate_group",
            smarts_pattern="[#15](=O)([OX2H1,OX1-])[OX2H1,OX1-]",
            target=DrugTarget(
                target_id="TGT_NUCLEOTIDE_ENZYME",
                name="Nucleotide-processing enzyme",
                target_type=TargetType.ENZYME,
                organism="Various",
                description=(
                    "Phosphonate groups often mimic phosphate esters found "
                    "in nucleotide substrates, suggesting interaction with "
                    "nucleotide-processing enzymes."
                ),
            ),
            base_confidence=0.45,
            description="Phosphonate functional group detected.",
        ),
    ]


# ---------------------------------------------------------------------------
# Prediction engine interface
# ---------------------------------------------------------------------------


class TargetPredictionEngine(ABC):
    """
    Abstract interface for compound-target prediction backends.

    Any prediction backend -- rule-based, classical machine learning, or
    deep learning -- must implement this interface so that it can be
    used interchangeably by ``TargetPredictionService`` and, by
    extension, ``pipeline.py``.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """
        Return a short, human-readable identifier for this engine.

        Returns:
            The engine's name (e.g., "rule_based_v1", "gnn_target_v2").
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: MolecularFeatures) -> list[TargetPrediction]:
        """
        Predict candidate targets for a single compound.

        Args:
            features: The ``MolecularFeatures`` representation of the
                compound to evaluate.

        Returns:
            A list of ``TargetPrediction`` instances. Ordering is not
            guaranteed; callers should sort results as needed.

        Raises:
            TargetPredictionError: If prediction fails unexpectedly.
        """
        raise NotImplementedError


class RuleBasedTargetPredictor(TargetPredictionEngine):
    """
    Baseline rule-based target prediction engine.

    Predicts candidate targets by matching SMARTS substructure patterns,
    drawn from a curated knowledge base, against the query molecule. This
    engine requires no trained model and is intended to keep the
    ``target_prediction`` module fully functional out of the box.

    The confidence score for a matching rule scales with the number of
    non-overlapping substructure matches found, capped at 1.0, to
    slightly favor compounds exhibiting a motif more prominently while
    still being driven primarily by the rule's base confidence.

    Attributes:
        rules: The list of ``TargetPredictionRule`` instances used for
            matching.
    """

    _ENGINE_NAME: str = "rule_based_v1"
    _MATCH_COUNT_BONUS_PER_EXTRA_MATCH: float = 0.05

    def __init__(self, rules: list[TargetPredictionRule] | None = None) -> None:
        """
        Initialize the rule-based predictor.

        Args:
            rules: Optional custom list of ``TargetPredictionRule``
                instances. If omitted, the default curated knowledge base
                is used.
        """
        self.rules: list[TargetPredictionRule] = (
            rules if rules is not None else _build_default_rule_knowledge_base()
        )
        self._compiled_patterns: dict[str, Chem.Mol] = {
            rule.rule_id: Chem.MolFromSmarts(rule.smarts_pattern)
            for rule in self.rules
        }

    @property
    def engine_name(self) -> str:
        """Return the engine's identifier."""
        return self._ENGINE_NAME

    @staticmethod
    def _confidence_to_level(confidence_score: float) -> PredictionConfidence:
        """
        Map a numeric confidence score to a qualitative confidence
        bucket.

        Args:
            confidence_score: Numeric confidence score, in [0.0, 1.0].

        Returns:
            The corresponding ``PredictionConfidence`` bucket.
        """
        if confidence_score >= 0.85:
            return PredictionConfidence.VERY_HIGH
        if confidence_score >= 0.65:
            return PredictionConfidence.HIGH
        if confidence_score >= 0.4:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def predict(self, features: MolecularFeatures) -> list[TargetPrediction]:
        """
        Predict candidate targets for a compound by matching structural
        rules against its molecule.

        Args:
            features: The ``MolecularFeatures`` representation of the
                compound to evaluate.

        Returns:
            A list of ``TargetPrediction`` instances, one per matching
            rule. The list is unsorted; callers should sort by
            ``confidence_score`` as needed.

        Raises:
            TargetPredictionError: If substructure matching fails
                unexpectedly for a rule.
        """
        predictions: list[TargetPrediction] = []

        for rule in self.rules:
            pattern = self._compiled_patterns.get(rule.rule_id)
            if pattern is None:
                logger.warning(
                    "Skipping rule '%s' with an uncompiled SMARTS pattern.",
                    rule.rule_id,
                )
                continue

            try:
                matches = features.mol.GetSubstructMatches(pattern, uniquify=True)
            except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
                logger.exception(
                    "Substructure matching failed for rule '%s' on "
                    "compound_id='%s'.",
                    rule.rule_id,
                    features.compound_id,
                )
                raise TargetPredictionError(
                    compound_id=features.compound_id,
                    engine_name=self.engine_name,
                    message=(
                        f"Substructure matching failed for rule "
                        f"'{rule.rule_id}'."
                    ),
                ) from exc

            match_count = len(matches)
            if match_count == 0:
                continue

            confidence_score = min(
                1.0,
                rule.base_confidence
                + (match_count - 1) * self._MATCH_COUNT_BONUS_PER_EXTRA_MATCH,
            )

            logger.debug(
                "Rule '%s' matched %d time(s) for compound_id='%s' "
                "(confidence=%.3f).",
                rule.rule_id,
                match_count,
                features.compound_id,
                confidence_score,
            )

            predictions.append(
                TargetPrediction(
                    compound_id=features.compound_id,
                    target=rule.target,
                    confidence_score=confidence_score,
                    confidence_level=self._confidence_to_level(confidence_score),
                    prediction_method=self.engine_name,
                )
            )

        return predictions


# ---------------------------------------------------------------------------
# Prediction service (orchestration layer)
# ---------------------------------------------------------------------------


class TargetPredictionService:
    """
    Orchestrates SMILES validation, feature extraction, and target
    prediction using a pluggable ``TargetPredictionEngine``.

    This is the primary integration point for ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``. The prediction backend can be
    swapped (e.g., for a trained machine learning or deep learning
    model) by constructing this service with a different
    ``TargetPredictionEngine`` implementation, without changing any
    calling code.

    Attributes:
        engine: The active ``TargetPredictionEngine`` used for
            prediction.
        fingerprint_radius: Morgan fingerprint radius used for feature
            extraction.
        fingerprint_n_bits: Morgan fingerprint bit-vector length used for
            feature extraction.
    """

    def __init__(
        self,
        engine: TargetPredictionEngine | None = None,
        fingerprint_radius: int = 2,
        fingerprint_n_bits: int = 2048,
    ) -> None:
        """
        Initialize the target prediction service.

        Args:
            engine: The prediction engine to use. If omitted, a
                ``RuleBasedTargetPredictor`` baseline is used.
            fingerprint_radius: Morgan fingerprint radius used when
                extracting features (default 2, i.e. ECFP4).
            fingerprint_n_bits: Morgan fingerprint bit-vector length used
                when extracting features (default 2048).
        """
        self.engine: TargetPredictionEngine = engine or RuleBasedTargetPredictor()
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

    def predict_targets(
        self,
        smiles: str,
        compound_id: str,
        top_k: int | None = None,
        min_confidence: float | None = None,
    ) -> list[TargetPrediction]:
        """
        Predict and rank candidate targets for a single compound.

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound being evaluated.
            top_k: If provided, limits the number of returned predictions
                to the top ``top_k`` highest-confidence targets.
            min_confidence: If provided, filters out predictions with a
                ``confidence_score`` below this threshold, in
                [0.0, 1.0].

        Returns:
            A list of ``TargetPrediction`` instances sorted by
            ``confidence_score`` in descending order.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly.
            TargetPredictionError: If the underlying engine fails
                unexpectedly during prediction.
            ValueError: If ``top_k`` or ``min_confidence`` are out of
                valid range.
        """
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be a positive integer when provided.")
        if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0.0, 1.0].")

        logger.info(
            "Predicting targets for compound_id='%s' using engine '%s'.",
            compound_id,
            self.engine.engine_name,
        )

        features = self.extract_features(smiles, compound_id)

        try:
            predictions = self.engine.predict(features)
        except TargetPredictionError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Unexpected engine failure for compound_id='%s'.", compound_id
            )
            raise TargetPredictionError(
                compound_id=compound_id, engine_name=self.engine.engine_name
            ) from exc

        if min_confidence is not None:
            predictions = [
                prediction
                for prediction in predictions
                if prediction.confidence_score >= min_confidence
            ]

        predictions.sort(key=lambda prediction: prediction.confidence_score, reverse=True)

        if top_k is not None:
            predictions = predictions[:top_k]

        logger.info(
            "Predicted %d target(s) for compound_id='%s'.",
            len(predictions),
            compound_id,
        )
        return predictions

    def predict_targets_batch(
        self,
        compounds: list[tuple[str, str]],
        top_k: int | None = None,
        min_confidence: float | None = None,
        skip_invalid: bool = True,
    ) -> dict[str, list[TargetPrediction]]:
        """
        Predict and rank candidate targets for multiple compounds.

        Args:
            compounds: A list of ``(smiles, compound_id)`` tuples to
                evaluate.
            top_k: If provided, limits each compound's returned
                predictions to the top ``top_k`` highest-confidence
                targets.
            min_confidence: If provided, filters out predictions with a
                ``confidence_score`` below this threshold, in
                [0.0, 1.0], for every compound.
            skip_invalid: If True, compounds with unparseable SMILES are
                logged and skipped rather than raising an exception. If
                False, an invalid SMILES causes ``InvalidSMILESError`` to
                propagate.

        Returns:
            A mapping of ``compound_id`` to its ranked list of
            ``TargetPrediction`` instances.

        Raises:
            InvalidSMILESError: If a compound's SMILES cannot be parsed
                and ``skip_invalid`` is False.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly for a compound.
            TargetPredictionError: If the underlying engine fails
                unexpectedly during prediction for a compound.
        """
        results: dict[str, list[TargetPrediction]] = {}

        for smiles, compound_id in compounds:
            try:
                results[compound_id] = self.predict_targets(
                    smiles=smiles,
                    compound_id=compound_id,
                    top_k=top_k,
                    min_confidence=min_confidence,
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

_default_service = TargetPredictionService()


def predict_targets(
    smiles: str,
    compound_id: str,
    top_k: int | None = None,
    min_confidence: float | None = None,
) -> list[TargetPrediction]:
    """
    Predict and rank candidate targets for a single compound using the
    default rule-based prediction service.

    This is the primary entry point intended for use by ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``.

    Args:
        smiles: The SMILES string representing the compound.
        compound_id: Identifier of the compound being evaluated.
        top_k: If provided, limits the number of returned predictions to
            the top ``top_k`` highest-confidence targets.
        min_confidence: If provided, filters out predictions with a
            ``confidence_score`` below this threshold, in [0.0, 1.0].

    Returns:
        A list of ``TargetPrediction`` instances sorted by
        ``confidence_score`` in descending order.

    Raises:
        InvalidSMILESError: If the SMILES string cannot be parsed.
        DescriptorCalculationError: If fingerprint generation fails
            unexpectedly.
        TargetPredictionError: If the underlying engine fails
            unexpectedly during prediction.
    """
    return _default_service.predict_targets(
        smiles=smiles,
        compound_id=compound_id,
        top_k=top_k,
        min_confidence=min_confidence,
    )


def predict_targets_batch(
    compounds: list[tuple[str, str]],
    top_k: int | None = None,
    min_confidence: float | None = None,
    skip_invalid: bool = True,
) -> dict[str, list[TargetPrediction]]:
    """
    Predict and rank candidate targets for multiple compounds using the
    default rule-based prediction service.

    Args:
        compounds: A list of ``(smiles, compound_id)`` tuples to
            evaluate.
        top_k: If provided, limits each compound's returned predictions
            to the top ``top_k`` highest-confidence targets.
        min_confidence: If provided, filters out predictions with a
            ``confidence_score`` below this threshold, in [0.0, 1.0],
            for every compound.
        skip_invalid: If True, compounds with unparseable SMILES are
            logged and skipped rather than raising an exception.

    Returns:
        A mapping of ``compound_id`` to its ranked list of
        ``TargetPrediction`` instances.

    Raises:
        InvalidSMILESError: If a compound's SMILES cannot be parsed and
            ``skip_invalid`` is False.
        DescriptorCalculationError: If fingerprint generation fails
            unexpectedly for a compound.
        TargetPredictionError: If the underlying engine fails
            unexpectedly during prediction for a compound.
    """
    return _default_service.predict_targets_batch(
        compounds=compounds,
        top_k=top_k,
        min_confidence=min_confidence,
        skip_invalid=skip_invalid,
    )