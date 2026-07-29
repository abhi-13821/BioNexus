"""
drug_discovery/molecular_properties.py

RDKit-backed molecular property calculation for the BioNexus Drug
Discovery module.

This module is responsible for:
    - Validating SMILES strings.
    - Converting SMILES strings into RDKit molecule objects.
    - Computing common physicochemical descriptors (molecular weight,
      LogP, TPSA, H-bond donors/acceptors, rotatable bonds, heavy atom
      count, ring counts, aromatic ring count, Fraction Csp3, exact
      molecular weight, and QED).
    - Evaluating Lipinski's Rule of Five compliance.

The module exposes both a class-based API (``MolecularPropertyCalculator``)
and thin module-level convenience functions so it can be consumed
uniformly by ``pipeline.py``, ``drug_info``, and ``smiles_analyzer``.

All computed core descriptors are returned using the ``MolecularProperties``
dataclass defined in ``drug_discovery.models``. Descriptors that fall
outside the schema of ``MolecularProperties`` (ring count, Fraction Csp3,
exact molecular weight) are returned via the supplementary
``ExtendedMolecularDescriptors`` dataclass defined in this module.

Compatibility
-------------
Targets Python 3.11 and RDKit (rdkit-pypi / rdkit package). No mutation of
``drug_discovery.models`` is performed or required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from drug_discovery.models import MolecularProperties

__all__ = [
    "InvalidSMILESError",
    "DescriptorCalculationError",
    "ExtendedMolecularDescriptors",
    "MolecularPropertyCalculator",
    "validate_smiles",
    "calculate_molecular_properties",
    "calculate_extended_descriptors",
]

logger = logging.getLogger(__name__)

# Silence RDKit's native C++ logger so parsing errors are handled solely
# through this module's exceptions and logging, rather than being printed
# directly to stderr by RDKit itself.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class InvalidSMILESError(ValueError):
    """
    Raised when a SMILES string cannot be parsed into a valid RDKit
    molecule.

    Attributes:
        smiles: The offending SMILES string that failed validation or
            parsing.
    """

    def __init__(self, smiles: str, message: str | None = None) -> None:
        self.smiles = smiles
        resolved_message = message or f"Invalid SMILES string: '{smiles}'"
        super().__init__(resolved_message)


class DescriptorCalculationError(RuntimeError):
    """
    Raised when descriptor calculation fails for an otherwise valid RDKit
    molecule (e.g., due to an unexpected RDKit internal error).

    Attributes:
        smiles: The SMILES string associated with the failed calculation.
    """

    def __init__(self, smiles: str, message: str | None = None) -> None:
        self.smiles = smiles
        resolved_message = (
            message or f"Failed to calculate descriptors for SMILES: '{smiles}'"
        )
        super().__init__(resolved_message)


# ---------------------------------------------------------------------------
# Supplementary descriptor model
# ---------------------------------------------------------------------------


@dataclass
class ExtendedMolecularDescriptors:
    """
    Supplementary molecular descriptors that fall outside the schema of
    ``drug_discovery.models.MolecularProperties``.

    This dataclass is intentionally local to ``molecular_properties.py``
    so that ``drug_discovery.models`` remains unmodified. Consumers that
    need the full descriptor set (e.g., ``smiles_analyzer``) should use
    this in conjunction with ``MolecularProperties``.

    Attributes:
        compound_id: Identifier of the compound these descriptors belong
            to.
        exact_molecular_weight: Monoisotopic (exact) molecular weight,
            in g/mol.
        ring_count: Total number of rings (aromatic and non-aromatic).
        fraction_csp3: Fraction of carbons that are sp3-hybridized,
            in [0.0, 1.0].
        qed_score: Quantitative Estimate of Drug-likeness, in [0.0, 1.0].
    """

    compound_id: str
    exact_molecular_weight: float
    ring_count: int
    fraction_csp3: float
    qed_score: float

    def __post_init__(self) -> None:
        """Validate range constraints on the supplementary descriptors."""
        if not self.compound_id.strip():
            raise ValueError("compound_id must be a non-empty string.")
        if self.exact_molecular_weight < 0:
            raise ValueError("exact_molecular_weight cannot be negative.")
        if self.ring_count < 0:
            raise ValueError("ring_count cannot be negative.")
        if not 0.0 <= self.fraction_csp3 <= 1.0:
            raise ValueError("fraction_csp3 must be within [0.0, 1.0].")
        if not 0.0 <= self.qed_score <= 1.0:
            raise ValueError("qed_score must be within [0.0, 1.0].")


# ---------------------------------------------------------------------------
# Lipinski Rule of Five thresholds
# ---------------------------------------------------------------------------

_LIPINSKI_MAX_MOLECULAR_WEIGHT: float = 500.0
_LIPINSKI_MAX_LOGP: float = 5.0
_LIPINSKI_MAX_H_BOND_DONORS: int = 5
_LIPINSKI_MAX_H_BOND_ACCEPTORS: int = 10


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class MolecularPropertyCalculator:
    """
    Computes molecular descriptors and Lipinski compliance for compounds
    represented as SMILES strings, using RDKit.

    This class is stateless with respect to compound data; each method
    call operates independently on the SMILES/molecule provided. It is
    safe to instantiate a single shared instance and reuse it across
    ``pipeline.py``, ``drug_info``, and ``smiles_analyzer``.
    """

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """
        Validate whether a SMILES string can be parsed into a valid RDKit
        molecule.

        Args:
            smiles: The SMILES string to validate.

        Returns:
            True if the SMILES string is syntactically and semantically
            valid according to RDKit, False otherwise.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            logger.debug("Rejected empty or non-string SMILES input.")
            return False

        mol = Chem.MolFromSmiles(smiles.strip())
        is_valid = mol is not None
        if not is_valid:
            logger.debug("RDKit failed to parse SMILES: '%s'", smiles)
        return is_valid

    @staticmethod
    def parse_smiles(smiles: str) -> Chem.Mol:
        """
        Parse a SMILES string into an RDKit molecule object.

        Args:
            smiles: The SMILES string to parse.

        Returns:
            A valid RDKit ``Mol`` object.

        Raises:
            InvalidSMILESError: If the SMILES string is empty, not a
                string, or cannot be parsed by RDKit.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            raise InvalidSMILESError(
                smiles if isinstance(smiles, str) else str(smiles),
                message="SMILES input must be a non-empty string.",
            )

        cleaned_smiles = smiles.strip()
        mol = Chem.MolFromSmiles(cleaned_smiles)
        if mol is None:
            raise InvalidSMILESError(cleaned_smiles)

        logger.debug("Successfully parsed SMILES: '%s'", cleaned_smiles)
        return mol

    @staticmethod
    def calculate_lipinski_violations(
        molecular_weight: float,
        logp: float,
        h_bond_donors: int,
        h_bond_acceptors: int,
    ) -> int:
        """
        Determine the number of Lipinski's Rule of Five violations.

        The four standard criteria evaluated are:
            - Molecular weight <= 500 g/mol
            - LogP <= 5
            - H-bond donors <= 5
            - H-bond acceptors <= 10

        Args:
            molecular_weight: Molecular weight in g/mol.
            logp: Calculated octanol-water partition coefficient.
            h_bond_donors: Count of hydrogen bond donors.
            h_bond_acceptors: Count of hydrogen bond acceptors.

        Returns:
            An integer count of violated criteria, in the range [0, 4].
        """
        violations = 0
        if molecular_weight > _LIPINSKI_MAX_MOLECULAR_WEIGHT:
            violations += 1
        if logp > _LIPINSKI_MAX_LOGP:
            violations += 1
        if h_bond_donors > _LIPINSKI_MAX_H_BOND_DONORS:
            violations += 1
        if h_bond_acceptors > _LIPINSKI_MAX_H_BOND_ACCEPTORS:
            violations += 1
        return violations

    def calculate(self, smiles: str, compound_id: str) -> MolecularProperties:
        """
        Compute core molecular descriptors for a compound and return them
        as a ``MolecularProperties`` instance.

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound the descriptors belong
                to.

        Returns:
            A populated ``MolecularProperties`` instance.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If descriptor calculation fails
                unexpectedly for a successfully parsed molecule.
        """
        mol = self.parse_smiles(smiles)

        try:
            molecular_weight = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            h_bond_donors = Lipinski.NumHDonors(mol)
            h_bond_acceptors = Lipinski.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            heavy_atom_count = mol.GetNumHeavyAtoms()
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            qed_score = QED.qed(mol)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Descriptor calculation failed for SMILES: '%s'", smiles
            )
            raise DescriptorCalculationError(smiles) from exc

        lipinski_violations = self.calculate_lipinski_violations(
            molecular_weight=molecular_weight,
            logp=logp,
            h_bond_donors=h_bond_donors,
            h_bond_acceptors=h_bond_acceptors,
        )

        logger.info(
            "Computed molecular properties for compound_id='%s' "
            "(MW=%.2f, LogP=%.2f, violations=%d)",
            compound_id,
            molecular_weight,
            logp,
            lipinski_violations,
        )

        return MolecularProperties(
            compound_id=compound_id,
            molecular_weight=molecular_weight,
            logp=logp,
            tpsa=tpsa,
            h_bond_donors=h_bond_donors,
            h_bond_acceptors=h_bond_acceptors,
            rotatable_bonds=rotatable_bonds,
            aromatic_rings=aromatic_rings,
            heavy_atom_count=heavy_atom_count,
            lipinski_violations=lipinski_violations,
            qed_score=qed_score,
        )

    def calculate_extended(
        self, smiles: str, compound_id: str
    ) -> ExtendedMolecularDescriptors:
        """
        Compute supplementary molecular descriptors not covered by
        ``MolecularProperties`` (exact molecular weight, total ring count,
        Fraction Csp3, and QED score).

        Args:
            smiles: The SMILES string representing the compound.
            compound_id: Identifier of the compound the descriptors belong
                to.

        Returns:
            A populated ``ExtendedMolecularDescriptors`` instance.

        Raises:
            InvalidSMILESError: If the SMILES string cannot be parsed.
            DescriptorCalculationError: If descriptor calculation fails
                unexpectedly for a successfully parsed molecule.
        """
        mol = self.parse_smiles(smiles)

        try:
            exact_molecular_weight = Descriptors.ExactMolWt(mol)
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            qed_score = QED.qed(mol)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.exception(
                "Extended descriptor calculation failed for SMILES: '%s'",
                smiles,
            )
            raise DescriptorCalculationError(smiles) from exc

        logger.info(
            "Computed extended molecular descriptors for compound_id='%s' "
            "(exact_MW=%.4f, rings=%d, fraction_csp3=%.3f)",
            compound_id,
            exact_molecular_weight,
            ring_count,
            fraction_csp3,
        )

        return ExtendedMolecularDescriptors(
            compound_id=compound_id,
            exact_molecular_weight=exact_molecular_weight,
            ring_count=ring_count,
            fraction_csp3=fraction_csp3,
            qed_score=qed_score,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_calculator = MolecularPropertyCalculator()


def validate_smiles(smiles: str) -> bool:
    """
    Validate whether a SMILES string represents a chemically valid
    structure.

    Args:
        smiles: The SMILES string to validate.

    Returns:
        True if the SMILES string is valid, False otherwise.
    """
    return _default_calculator.validate_smiles(smiles)


def calculate_molecular_properties(
    smiles: str, compound_id: str
) -> MolecularProperties:
    """
    Compute core molecular descriptors for a compound.

    This is the primary entry point intended for use by ``pipeline.py``,
    ``drug_info``, and ``smiles_analyzer``.

    Args:
        smiles: The SMILES string representing the compound.
        compound_id: Identifier of the compound the descriptors belong to.

    Returns:
        A populated ``MolecularProperties`` instance.

    Raises:
        InvalidSMILESError: If the SMILES string cannot be parsed.
        DescriptorCalculationError: If descriptor calculation fails
            unexpectedly for a successfully parsed molecule.
    """
    return _default_calculator.calculate(smiles=smiles, compound_id=compound_id)


def calculate_extended_descriptors(
    smiles: str, compound_id: str
) -> ExtendedMolecularDescriptors:
    """
    Compute supplementary molecular descriptors (exact molecular weight,
    ring count, Fraction Csp3, QED score) for a compound.

    Args:
        smiles: The SMILES string representing the compound.
        compound_id: Identifier of the compound the descriptors belong to.

    Returns:
        A populated ``ExtendedMolecularDescriptors`` instance.

    Raises:
        InvalidSMILESError: If the SMILES string cannot be parsed.
        DescriptorCalculationError: If descriptor calculation fails
            unexpectedly for a successfully parsed molecule.
    """
    return _default_calculator.calculate_extended(
        smiles=smiles, compound_id=compound_id
    )