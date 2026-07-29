"""
drug_discovery/compound_similarity.py

RDKit-backed compound similarity search for the BioNexus Drug Discovery
module.

This module is responsible for:
    - Validating SMILES strings (delegated to
      ``drug_discovery.molecular_properties``).
    - Generating Morgan (ECFP4) fingerprints for compounds.
    - Computing Tanimoto similarity between fingerprints.
    - Comparing a single query compound against a collection of candidate
      compounds and returning ranked, threshold-filtered results.

The module exposes both a class-based API
(``CompoundSimilaritySearchEngine``) and thin module-level convenience
functions so it can be consumed uniformly by ``pipeline.py``,
``drug_info``, and ``smiles_analyzer``.

Results are returned using the ``SimilarCompound`` and ``Compound``
dataclasses defined in ``drug_discovery.models``. SMILES validation and
parsing reuse the implementation in
``drug_discovery.molecular_properties`` to avoid duplicating RDKit
parsing logic, and the same ``InvalidSMILESError`` exception is raised
for invalid input so callers only need to handle one exception type
across both modules.

Compatibility
-------------
Targets Python 3.11 and RDKit (rdkit-pypi / rdkit package).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from drug_discovery.models import Compound, SimilarCompound, SimilarityMetric
from drug_discovery.molecular_properties import (
    DescriptorCalculationError,
    InvalidSMILESError,
    MolecularPropertyCalculator,
)

__all__ = [
    "InvalidSMILESError",
    "DescriptorCalculationError",
    "FingerprintConfig",
    "CompoundSimilaritySearchEngine",
    "generate_morgan_fingerprint",
    "compute_tanimoto_similarity",
    "find_similar_compounds",
]

logger = logging.getLogger(__name__)

# Default Morgan fingerprint parameters. radius=2 with a folded bit vector
# is the standard configuration equivalent to ECFP4 (the "4" refers to the
# diameter of the considered atom environment, i.e. 2 * radius).
_DEFAULT_MORGAN_RADIUS: int = 2
_DEFAULT_MORGAN_N_BITS: int = 2048


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerprintConfig:
    """
    Configuration governing Morgan fingerprint generation and similarity
    filtering.

    Attributes:
        radius: Morgan fingerprint radius. A radius of 2 corresponds to
            ECFP4 (atom environments of diameter 4).
        n_bits: Length of the folded fingerprint bit vector.
        use_chirality: Whether to encode chirality information in the
            fingerprint.
        use_features: Whether to generate FCFP-style (feature-based)
            fingerprints instead of standard ECFP fingerprints.
        similarity_threshold: Minimum Tanimoto similarity score, in
            [0.0, 1.0], required for a candidate to be included in
            search results.
    """

    radius: int = _DEFAULT_MORGAN_RADIUS
    n_bits: int = _DEFAULT_MORGAN_N_BITS
    use_chirality: bool = False
    use_features: bool = False
    similarity_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.radius < 0:
            raise ValueError("radius cannot be negative.")
        if self.n_bits < 1:
            raise ValueError("n_bits must be a positive integer.")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be within [0.0, 1.0]."
            )


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------


class CompoundSimilaritySearchEngine:
    """
    Performs Morgan-fingerprint-based Tanimoto similarity search of a
    query compound against a collection of candidate compounds.

    This class is stateless with respect to compound data beyond its
    fingerprinting configuration; a single instance may be safely reused
    across ``pipeline.py``, ``drug_info``, and ``smiles_analyzer``.

    Attributes:
        config: The ``FingerprintConfig`` governing fingerprint generation
            and default similarity filtering.
    """

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        """
        Initialize the similarity search engine.

        Args:
            config: Optional fingerprint configuration. If omitted, a
                default ``FingerprintConfig`` (ECFP4, 2048 bits, no
                similarity threshold) is used.
        """
        self.config = config or FingerprintConfig()
        self._property_calculator = MolecularPropertyCalculator()

    def _parse(self, smiles: str) -> Chem.Mol:
        """
        Parse and validate a SMILES string into an RDKit molecule.

        Args:
            smiles: The SMILES string to parse.

        Returns:
            A valid RDKit ``Mol`` object.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
        """
        return self._property_calculator.parse_smiles(smiles)

    def generate_fingerprint(self, smiles: str) -> ExplicitBitVect:
        """
        Generate a Morgan (ECFP4-equivalent) fingerprint for a SMILES
        string.

        Args:
            smiles: The SMILES string representing the compound.

        Returns:
            A folded RDKit ``ExplicitBitVect`` fingerprint.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly for a successfully parsed molecule.
        """
        mol = self._parse(smiles)
        return self._generate_fingerprint_from_mol(mol, smiles)

    def _generate_fingerprint_from_mol(
        self, mol: Chem.Mol, smiles: str
    ) -> ExplicitBitVect:
        """
        Generate a Morgan fingerprint from an already-parsed RDKit
        molecule.

        Args:
            mol: A valid RDKit ``Mol`` object.
            smiles: The original SMILES string, used only for error
                reporting and logging.

        Returns:
            A folded RDKit ``ExplicitBitVect`` fingerprint.

        Raises:
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly.
        """
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.config.radius,
                nBits=self.config.n_bits,
                useChirality=self.config.use_chirality,
                useFeatures=self.config.use_features,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Fingerprint generation failed for SMILES: '%s'", smiles
            )
            raise DescriptorCalculationError(smiles) from exc

        return fingerprint

    def compute_similarity(
        self, fingerprint_a: ExplicitBitVect, fingerprint_b: ExplicitBitVect
    ) -> float:
        """
        Compute the Tanimoto similarity between two fingerprints.

        Args:
            fingerprint_a: The first fingerprint.
            fingerprint_b: The second fingerprint.

        Returns:
            The Tanimoto similarity coefficient, in [0.0, 1.0].
        """
        return DataStructs.TanimotoSimilarity(fingerprint_a, fingerprint_b)

    def compute_pairwise_similarity(
        self, smiles_a: str, smiles_b: str
    ) -> float:
        """
        Compute the Tanimoto similarity between two compounds given
        directly as SMILES strings.

        Args:
            smiles_a: SMILES string of the first compound.
            smiles_b: SMILES string of the second compound.

        Returns:
            The Tanimoto similarity coefficient, in [0.0, 1.0].

        Raises:
            InvalidSMILESError: If either SMILES string cannot be parsed.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly for either compound.
        """
        fingerprint_a = self.generate_fingerprint(smiles_a)
        fingerprint_b = self.generate_fingerprint(smiles_b)
        return self.compute_similarity(fingerprint_a, fingerprint_b)

    def search(
        self,
        query_smiles: str,
        candidates: list[Compound],
        threshold: float | None = None,
        top_k: int | None = None,
        skip_invalid_candidates: bool = True,
    ) -> list[SimilarCompound]:
        """
        Compare a query compound against multiple candidate compounds and
        return ranked, threshold-filtered similarity results.

        Args:
            query_smiles: SMILES string of the query compound.
            candidates: List of candidate ``Compound`` instances to
                compare against the query.
            threshold: Minimum Tanimoto similarity score, in [0.0, 1.0],
                required for a candidate to be included in the results.
                If omitted, ``self.config.similarity_threshold`` is used.
            top_k: If provided, limits the number of returned results to
                the top ``top_k`` highest-scoring candidates.
            skip_invalid_candidates: If True, candidates with unparseable
                SMILES are logged and skipped rather than raising an
                exception. If False, an invalid candidate SMILES causes
                ``InvalidSMILESError`` to propagate.

        Returns:
            A list of ``SimilarCompound`` instances sorted by similarity
            score in descending order, with ``rank`` populated
            (1-based).

        Raises:
            InvalidSMILESError: If the query SMILES cannot be parsed, or
                if a candidate SMILES cannot be parsed and
                ``skip_invalid_candidates`` is False.
            DescriptorCalculationError: If fingerprint generation fails
                unexpectedly for the query or a candidate.
        """
        effective_threshold = (
            threshold if threshold is not None else self.config.similarity_threshold
        )
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0].")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be a positive integer when provided.")

        query_fingerprint = self.generate_fingerprint(query_smiles)
        logger.info(
            "Starting similarity search against %d candidate(s) "
            "(threshold=%.3f, top_k=%s).",
            len(candidates),
            effective_threshold,
            top_k,
        )

        scored_results: list[SimilarCompound] = []
        for candidate in candidates:
            try:
                candidate_fingerprint = self.generate_fingerprint(candidate.smiles)
            except InvalidSMILESError:
                if skip_invalid_candidates:
                    logger.warning(
                        "Skipping candidate '%s' with invalid SMILES: '%s'",
                        candidate.compound_id,
                        candidate.smiles,
                    )
                    continue
                raise

            similarity_score = self.compute_similarity(
                query_fingerprint, candidate_fingerprint
            )
            if similarity_score >= effective_threshold:
                scored_results.append(
                    SimilarCompound(
                        compound=candidate,
                        similarity_score=similarity_score,
                        metric=SimilarityMetric.TANIMOTO,
                    )
                )

        scored_results.sort(key=lambda result: result.similarity_score, reverse=True)

        if top_k is not None:
            scored_results = scored_results[:top_k]

        ranked_results = [
            SimilarCompound(
                compound=result.compound,
                similarity_score=result.similarity_score,
                metric=result.metric,
                rank=rank,
            )
            for rank, result in enumerate(scored_results, start=1)
        ]

        logger.info(
            "Similarity search complete: %d result(s) met threshold %.3f.",
            len(ranked_results),
            effective_threshold,
        )
        return ranked_results


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_engine = CompoundSimilaritySearchEngine()


def generate_morgan_fingerprint(
    smiles: str,
    radius: int = _DEFAULT_MORGAN_RADIUS,
    n_bits: int = _DEFAULT_MORGAN_N_BITS,
) -> ExplicitBitVect:
    """
    Generate a Morgan (ECFP4-equivalent, when radius=2) fingerprint for a
    SMILES string.

    Args:
        smiles: The SMILES string representing the compound.
        radius: Morgan fingerprint radius (default 2, equivalent to
            ECFP4).
        n_bits: Length of the folded fingerprint bit vector (default
            2048).

    Returns:
        A folded RDKit ``ExplicitBitVect`` fingerprint.

    Raises:
        InvalidSMILESError: If the SMILES string cannot be parsed.
        DescriptorCalculationError: If fingerprint generation fails
            unexpectedly for a successfully parsed molecule.
    """
    engine = CompoundSimilaritySearchEngine(
        config=FingerprintConfig(radius=radius, n_bits=n_bits)
    )
    return engine.generate_fingerprint(smiles)


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """
    Compute the Tanimoto similarity between two compounds given as SMILES
    strings, using default ECFP4 fingerprint parameters.

    Args:
        smiles_a: SMILES string of the first compound.
        smiles_b: SMILES string of the second compound.

    Returns:
        The Tanimoto similarity coefficient, in [0.0, 1.0].

    Raises:
        InvalidSMILESError: If either SMILES string cannot be parsed.
        DescriptorCalculationError: If fingerprint generation fails
            unexpectedly for either compound.
    """
    return _default_engine.compute_pairwise_similarity(smiles_a, smiles_b)


def find_similar_compounds(
    query_smiles: str,
    candidates: list[Compound],
    threshold: float = 0.0,
    top_k: int | None = None,
    skip_invalid_candidates: bool = True,
) -> list[SimilarCompound]:
    """
    Compare a query compound against multiple candidate compounds and
    return ranked, threshold-filtered similarity results, using default
    ECFP4 fingerprint parameters.

    This is the primary entry point intended for use by ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``.

    Args:
        query_smiles: SMILES string of the query compound.
        candidates: List of candidate ``Compound`` instances to compare
            against the query.
        threshold: Minimum Tanimoto similarity score, in [0.0, 1.0],
            required for a candidate to be included in the results.
        top_k: If provided, limits the number of returned results to the
            top ``top_k`` highest-scoring candidates.
        skip_invalid_candidates: If True, candidates with unparseable
            SMILES are logged and skipped rather than raising an
            exception.

    Returns:
        A list of ``SimilarCompound`` instances sorted by similarity
        score in descending order, with ``rank`` populated (1-based).

    Raises:
        InvalidSMILESError: If the query SMILES cannot be parsed, or if a
            candidate SMILES cannot be parsed and
            ``skip_invalid_candidates`` is False.
        DescriptorCalculationError: If fingerprint generation fails
            unexpectedly for the query or a candidate.
    """
    return _default_engine.search(
        query_smiles=query_smiles,
        candidates=candidates,
        threshold=threshold,
        top_k=top_k,
        skip_invalid_candidates=skip_invalid_candidates,
    )