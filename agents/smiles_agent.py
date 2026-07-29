"""
agents/smiles_agent.py

SMILES Agent for the BioNexus Multi-Agent AI System.

This agent specializes in chemical structure analysis from SMILES strings.
It wraps the existing smiles_analyzer module (frontend/smiles_analyzer.py)
and exposes its capabilities through the standardized BaseAgent interface.

The agent is responsible for:
    - Validating SMILES strings
    - Analyzing molecular structures
    - Computing molecular properties (MW, LogP, TPSA, etc.)
    - Detecting functional groups
    - Evaluating Lipinski's Rule of Five
    - Providing 3D structure information
    - Explaining molecular properties in user-friendly language
    - Maintaining conversation context through Memory
    - Returning structured responses compatible with agents.models

Design decisions:
    - The agent does NOT reimplement SMILES analysis logic; it delegates
      to the existing smiles_analyzer module's functions.
    - All heavy lifting (validation, property computation, 3D generation)
      is done by the smiles_analyzer module.
    - This agent only orchestrates: validates requests, calls the module,
      formats results, and manages memory.

Integration Points:
    - frontend.smiles_analyzer: validate_smiles, compute_properties,
      compute_advanced_descriptors, evaluate_lipinski,
      detect_functional_groups, generate_3d_molblock, build_atom_table
    - agents.base_agent: BaseAgent
    - agents.models: AgentRequest, AgentTask, AgentResult, AgentResponse
    - agents.memory: MemoryManager

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.base_agent import (
    AgentConfig,
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentNotInitializedError,
    AgentType,
    AgentValidationError,
    BaseAgent,
)
from agents.memory import MemoryManager
from agents.models import (
    AgentRequest,
    AgentResult,
    AgentTask,
    ConfidenceScore,
    TaskStatus,
)

# Import smiles_analyzer functionality with fallback
try:
    from frontend.smiles_analyzer import (
        ValidationResult,
        build_atom_table,
        compute_advanced_descriptors,
        compute_properties,
        detect_functional_groups,
        evaluate_lipinski,
        generate_3d_molblock,
        validate_smiles,
    )
except ImportError:
    # Fallback for testing/standalone usage
    ValidationResult = Any  # type: ignore
    build_atom_table = None  # type: ignore
    compute_advanced_descriptors = None  # type: ignore
    compute_properties = None  # type: ignore
    detect_functional_groups = None  # type: ignore
    evaluate_lipinski = None  # type: ignore
    generate_3d_molblock = None  # type: ignore
    validate_smiles = None  # type: ignore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# SmilesAgent
# ----------------------------------------------------------------------


class SmilesAgent(BaseAgent):
    """
    Agent specialized in chemical structure analysis from SMILES strings.

    This agent wraps the existing smiles_analyzer module and provides a
    standardized interface for the multi-agent system. It handles SMILES
    validation, molecular property computation, functional group detection,
    Lipinski rule evaluation, and 3D structure generation.

    Example:
        >>> agent = SmilesAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Analyze the molecule Aspirin",
        ...     parameters={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        ... )
        >>> response = await agent.handle_request(request)
        >>> print(response.message)

    Attributes:
        memory: MemoryManager instance for conversation history.
        include_3d: Whether to generate 3D structures for molecules.
        include_atom_table: Whether to include atom information tables.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        include_3d: bool = True,
        include_atom_table: bool = True,
    ) -> None:
        """
        Initialize the SMILES Agent.

        Args:
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default AgentConfig is used if not supplied.
            memory: MemoryManager instance for conversation history.
                A new InMemoryMemoryStore is used if not supplied.
            include_3d: Whether to generate 3D structures for molecules.
                Defaults to True.
            include_atom_table: Whether to include atom information tables.
                Defaults to True.
        """
        super().__init__(
            agent_type=AgentType.SMILES_ANALYSIS_AGENT,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=60.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=False,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.include_3d = include_3d
        self.include_atom_table = include_atom_table
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the SMILES Agent.

        This performs validation of the smiles_analyzer module availability
        and warms up any required resources. Called by BaseAgent.start().

        Raises:
            AgentInitializationError: If the smiles_analyzer module is not
                available or initialization fails.
        """
        if validate_smiles is None:
            raise AgentInitializationError(
                "SMILES analyzer module not available. "
                "Ensure frontend.smiles_analyzer is installed and accessible.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info(
            "SmilesAgent '%s' initialized with include_3d=%s, include_atom_table=%s",
            self.agent_id,
            self.include_3d,
            self.include_atom_table,
        )
        self._initialized = True

    def validate_request(self, request: AgentRequest) -> None:
        """
        Validate that the request is well-formed and executable by this agent.

        Args:
            request: The inbound request to validate.

        Raises:
            AgentValidationError: If the request is invalid.
        """
        if not request.instruction or not request.instruction.strip():
            raise AgentValidationError(
                "Instruction must be a non-empty string.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        if not self._initialized:
            raise AgentNotInitializedError(
                f"Agent '{self.agent_id}' ({self.agent_type.value}) "
                "received a request before start() was called.",
                agent_type=self.agent_type,
                is_recoverable=True,
            )

    def _extract_smiles_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract SMILES analysis parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of SMILES analysis parameters.
        """
        params = request.parameters or {}

        # Extract SMILES from parameters or instruction
        smiles = params.get("smiles", "")
        if not smiles and request.instruction:
            # Try to extract SMILES from instruction
            # Look for common SMILES patterns (letters, numbers, parentheses, brackets)
            import re
            smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]+'
            matches = re.findall(smiles_pattern, request.instruction)
            if matches:
                # Use the longest match as the SMILES
                smiles = max(matches, key=len)

        # Extract compound name
        compound_name = params.get("compound_name", "")

        # Action type: validate, analyze, properties, groups, lipinski, full
        action = params.get("action", "full")

        # Whether to include 3D structure
        include_3d = params.get("include_3d", self.include_3d)

        # Whether to include atom table
        include_atom_table = params.get("include_atom_table", self.include_atom_table)

        return {
            "smiles": smiles,
            "compound_name": compound_name,
            "action": action,
            "include_3d": include_3d,
            "include_atom_table": include_atom_table,
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the SMILES analysis task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the SMILES analysis results.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_smiles_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        action = params["action"]
        smiles = params["smiles"]

        if action == "validate":
            return await self._execute_validate(params, task)
        elif action == "properties":
            return await self._execute_properties(params, task)
        elif action == "groups":
            return await self._execute_functional_groups(params, task)
        elif action == "lipinski":
            return await self._execute_lipinski(params, task)
        else:
            # Default: full analysis
            return await self._execute_full_analysis(params, task)

    async def _execute_validate(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a SMILES validation request.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with validation results.

        Raises:
            AgentExecutionError: If validation fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for validation.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info("Validating SMILES: '%s'", smiles[:50])

        try:
            result = validate_smiles(smiles)

            output = {
                "smiles": smiles,
                "is_valid": result.is_valid,
                "canonical_smiles": result.canonical_smiles if result.is_valid else None,
                "error_message": result.error_message if not result.is_valid else None,
            }

            if result.is_valid:
                confidence = ConfidenceScore(
                    value=0.95,
                    basis=f"SMILES validated successfully: {result.canonical_smiles}",
                    source_module="smiles_analyzer",
                )
            else:
                confidence = ConfidenceScore(
                    value=0.9,
                    basis=f"SMILES validation failed: {result.error_message}",
                    source_module="smiles_analyzer",
                )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("SMILES validation failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"SMILES validation failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_properties(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a molecular properties request.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with molecular properties.

        Raises:
            AgentExecutionError: If property computation fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for property computation.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        # First validate the SMILES
        validation = validate_smiles(smiles)

        if not validation.is_valid:
            raise AgentExecutionError(
                f"Invalid SMILES: {validation.error_message}",
                agent_type=self.agent_type,
                is_recoverable=True,
            )

        canonical_smiles = validation.canonical_smiles

        try:
            props = compute_properties(canonical_smiles)

            # Get advanced descriptors if available
            advanced = {}
            if compute_advanced_descriptors:
                try:
                    advanced = compute_advanced_descriptors(canonical_smiles)
                except Exception as e:
                    self.logger.warning("Failed to compute advanced descriptors: %s", e)

            output = {
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "properties": props,
                "advanced_descriptors": advanced,
                "compound_name": params.get("compound_name", "Unnamed Compound"),
            }

            confidence = ConfidenceScore(
                value=0.9,
                basis=f"Properties computed for {canonical_smiles}",
                source_module="smiles_analyzer",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Property computation failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Property computation failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_functional_groups(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a functional group detection request.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with functional group detection results.

        Raises:
            AgentExecutionError: If functional group detection fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for functional group detection.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        # First validate the SMILES
        validation = validate_smiles(smiles)

        if not validation.is_valid:
            raise AgentExecutionError(
                f"Invalid SMILES: {validation.error_message}",
                agent_type=self.agent_type,
                is_recoverable=True,
            )

        canonical_smiles = validation.canonical_smiles

        try:
            groups = detect_functional_groups(canonical_smiles)

            output = {
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "functional_groups": groups,
                "compound_name": params.get("compound_name", "Unnamed Compound"),
            }

            confidence = ConfidenceScore(
                value=0.85,
                basis=f"Detected {len(groups)} functional groups",
                source_module="smiles_analyzer",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Functional group detection failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Functional group detection failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_lipinski(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a Lipinski Rule of Five evaluation request.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with Lipinski evaluation results.

        Raises:
            AgentExecutionError: If Lipinski evaluation fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for Lipinski evaluation.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        # First validate the SMILES
        validation = validate_smiles(smiles)

        if not validation.is_valid:
            raise AgentExecutionError(
                f"Invalid SMILES: {validation.error_message}",
                agent_type=self.agent_type,
                is_recoverable=True,
            )

        canonical_smiles = validation.canonical_smiles

        try:
            props = compute_properties(canonical_smiles)
            lipinski_report = evaluate_lipinski(props)

            output = {
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "lipinski": lipinski_report,
                "compound_name": params.get("compound_name", "Unnamed Compound"),
            }

            confidence = ConfidenceScore(
                value=0.85,
                basis=f"Lipinski evaluation: {lipinski_report.get('overall', 'unknown')}",
                source_module="smiles_analyzer",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Lipinski evaluation failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Lipinski evaluation failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_full_analysis(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a full SMILES analysis request.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with complete analysis results.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for full analysis.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info("Performing full SMILES analysis: '%s'", smiles[:50])

        try:
            # Validate SMILES
            validation = validate_smiles(smiles)

            if not validation.is_valid:
                raise AgentExecutionError(
                    f"Invalid SMILES: {validation.error_message}",
                    agent_type=self.agent_type,
                    is_recoverable=True,
                )

            canonical_smiles = validation.canonical_smiles

            # Compute properties
            props = compute_properties(canonical_smiles)

            # Compute advanced descriptors if available
            advanced = {}
            if compute_advanced_descriptors:
                try:
                    advanced = compute_advanced_descriptors(canonical_smiles)
                except Exception as e:
                    self.logger.warning("Failed to compute advanced descriptors: %s", e)

            # Detect functional groups
            groups = detect_functional_groups(canonical_smiles)

            # Evaluate Lipinski
            lipinski_report = evaluate_lipinski(props)

            # Generate 3D structure if requested
            molblock = None
            viewer_warning = None
            if self.include_3d and generate_3d_molblock:
                try:
                    molblock, viewer_warning = generate_3d_molblock(canonical_smiles)
                except Exception as e:
                    self.logger.warning("Failed to generate 3D structure: %s", e)
                    viewer_warning = str(e)

            # Build atom table if requested
            atom_table = None
            if self.include_atom_table and build_atom_table:
                try:
                    atom_table = build_atom_table(canonical_smiles)
                except Exception as e:
                    self.logger.warning("Failed to build atom table: %s", e)

            # Build output
            output = {
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "is_valid": True,
                "compound_name": params.get("compound_name", "Unnamed Compound"),
                "properties": props,
                "advanced_descriptors": advanced,
                "functional_groups": groups,
                "lipinski": lipinski_report,
                "has_3d": molblock is not None,
                "atom_count": len(atom_table) if atom_table else 0,
            }

            # Add 3D data if available
            if molblock:
                output["molblock"] = molblock
            if viewer_warning:
                output["viewer_warning"] = viewer_warning

            # Add atom table if available
            if atom_table:
                output["atom_table"] = atom_table

            # Add human-readable summary
            output["summary"] = self._generate_summary(
                params.get("compound_name", "Unnamed Compound"),
                canonical_smiles,
                props,
                groups,
                lipinski_report,
            )

            confidence = ConfidenceScore(
                value=0.9,
                basis=f"Full analysis completed for {canonical_smiles}",
                source_module="smiles_analyzer",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Full SMILES analysis failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Full SMILES analysis failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    def _generate_summary(
        self,
        name: str,
        smiles: str,
        props: dict[str, Any],
        groups: list[str],
        lipinski: dict[str, Any],
    ) -> str:
        """
        Generate a human-readable summary of the molecule.

        Args:
            name: The compound name.
            smiles: The canonical SMILES string.
            props: Molecular properties dictionary.
            groups: List of detected functional groups.
            lipinski: Lipinski evaluation report.

        Returns:
            A human-readable summary string.
        """
        summary_lines = [
            f"Compound: {name}",
            f"SMILES: {smiles}",
            "",
            "Molecular Properties:",
        ]

        # Add key properties
        key_props = [
            ("Molecular Formula", props.get("Formula", "N/A")),
            ("Exact Molecular Weight", f"{props.get('Exact Molecular Weight', 0):.2f} Da"),
            ("LogP", f"{props.get('LogP', 0):.2f}"),
            ("TPSA", f"{props.get('TPSA', 0):.2f}"),
            ("H-Bond Donors", str(props.get("H-Bond Donors", 0))),
            ("H-Bond Acceptors", str(props.get("H-Bond Acceptors", 0))),
            ("Rotatable Bonds", str(props.get("Rotatable Bonds", 0))),
        ]

        for key, value in key_props:
            summary_lines.append(f"  - {key}: {value}")

        # Add functional groups
        if groups:
            summary_lines.append("")
            summary_lines.append(f"Functional Groups detected: {', '.join(groups)}")
        else:
            summary_lines.append("")
            summary_lines.append("No common functional groups detected.")

        # Add Lipinski assessment
        if lipinski:
            summary_lines.append("")
            summary_lines.append(f"Lipinski Rule of Five: {lipinski.get('overall', 'unknown').upper()}")
            summary_lines.append(f"  Violations: {lipinski.get('violations', 0)}")

        return "\n".join(summary_lines)

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        This releases any resources acquired during initialization.
        """
        self.logger.info("Shutting down SmilesAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "smiles_validate",
            "smiles_properties",
            "smiles_functional_groups",
            "smiles_lipinski",
            "smiles_3d_structure",
            "smiles_full_analysis",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"SmilesAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_smiles_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    include_3d: bool = True,
    include_atom_table: bool = True,
) -> SmilesAgent:
    """
    Factory function to create a SmilesAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        include_3d: Whether to generate 3D structures for molecules.
        include_atom_table: Whether to include atom information tables.

    Returns:
        A configured SmilesAgent instance.
    """
    return SmilesAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        include_3d=include_3d,
        include_atom_table=include_atom_table,
    )


__all__: list[str] = [
    "SmilesAgent",
    "create_smiles_agent",
]