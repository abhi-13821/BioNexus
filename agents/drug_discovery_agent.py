"""
agents/drug_discovery_agent.py

Drug Discovery Agent for the BioNexus Multi-Agent AI System.

This agent specializes in AI-assisted drug discovery and candidate evaluation.
It wraps the existing drug_discovery pipeline (frontend/drug_discovery.py and
drug_discovery/pipeline.py) and exposes its capabilities through the
standardized BaseAgent interface.

The agent is responsible for:
    - Handling drug discovery and candidate evaluation queries
    - Supporting compound analysis using SMILES strings
    - Interpreting molecular properties in user-friendly language
    - Explaining compound similarity results
    - Explaining predicted biological targets
    - Explaining toxicity predictions
    - Explaining drug repurposing recommendations
    - Explaining candidate rankings
    - Generating concise, user-friendly summaries of pipeline output
    - Maintaining conversation context through Memory
    - Returning structured responses compatible with agents.models
    - Handling invalid compounds and pipeline failures gracefully
    - Supporting follow-up questions about previous analyses

Design decisions:
    - The agent does NOT reimplement drug discovery logic; it delegates
      to the existing drug_discovery pipeline and modules.
    - All heavy lifting (property calculation, similarity search,
      target prediction, toxicity prediction, repurposing, ranking)
      is done by the drug_discovery modules.
    - This agent only orchestrates: validates requests, calls the pipeline,
      formats results, and manages memory.

Integration Points:
    - drug_discovery.pipeline: run_drug_discovery, DrugDiscoveryPipeline,
      PipelineConfig
    - drug_discovery.models: Compound, DrugDiscoveryResult, DrugCandidate,
      CandidateRanking, MolecularProperties, TargetPrediction,
      ToxicityPrediction
    - drug_discovery.utils: formatting utilities
    - agents.base_agent: BaseAgent
    - agents.models: AgentRequest, AgentTask, AgentResult, AgentResponse
    - agents.memory: MemoryManager

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
import re
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

# Import drug_discovery functionality with fallback
try:
    from drug_discovery.models import (
        CandidateRanking,
        Compound,
        DrugCandidate,
        DrugDiscoveryResult,
        MolecularProperties,
        TargetPrediction,
        ToxicityPrediction,
    )
    from drug_discovery.pipeline import (
        PipelineConfig,
        run_drug_discovery,
    )
    from drug_discovery.utils import (
        format_molecular_weight,
        format_percentage,
        format_similarity_score,
    )
except ImportError:
    # Fallback for testing/standalone usage
    Compound = Any  # type: ignore
    DrugDiscoveryResult = Any  # type: ignore
    DrugCandidate = Any  # type: ignore
    CandidateRanking = Any  # type: ignore
    MolecularProperties = Any  # type: ignore
    TargetPrediction = Any  # type: ignore
    ToxicityPrediction = Any  # type: ignore
    run_drug_discovery = None  # type: ignore
    PipelineConfig = Any  # type: ignore
    format_molecular_weight = None  # type: ignore
    format_percentage = None  # type: ignore
    format_similarity_score = None  # type: ignore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DrugDiscoveryAgent
# ----------------------------------------------------------------------


class DrugDiscoveryAgent(BaseAgent):
    """
    Agent specialized in AI-assisted drug discovery and candidate evaluation.

    This agent wraps the existing drug_discovery pipeline and provides a
    standardized interface for the multi-agent system. It handles compound
    analysis, property interpretation, similarity search, target prediction,
    toxicity prediction, drug repurposing, and candidate ranking.

    Example:
        >>> agent = DrugDiscoveryAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Analyze Aspirin for drug discovery",
        ...     parameters={
        ...         "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        ...         "compound_name": "Aspirin"
        ...     }
        ... )
        >>> response = await agent.handle_request(request)
        >>> print(response.message)

    Attributes:
        memory: MemoryManager instance for conversation history.
        default_top_k: Default number of candidates to return.
        default_similarity_threshold: Default similarity threshold.
        include_all_stages: Whether to run all pipeline stages.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        default_top_k: int = 25,
        default_similarity_threshold: float = 0.7,
        include_all_stages: bool = True,
    ) -> None:
        """
        Initialize the Drug Discovery Agent.

        Args:
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default AgentConfig is used if not supplied.
            memory: MemoryManager instance for conversation history.
                A new InMemoryMemoryStore is used if not supplied.
            default_top_k: Default number of candidates to return.
                Defaults to 25.
            default_similarity_threshold: Default similarity threshold.
                Defaults to 0.7.
            include_all_stages: Whether to run all pipeline stages.
                Defaults to True.
        """
        super().__init__(
            agent_type=AgentType.DRUG_REPURPOSING_AGENT,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=120.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=False,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.default_top_k = default_top_k
        self.default_similarity_threshold = default_similarity_threshold
        self.include_all_stages = include_all_stages
        self._initialized = False
        self._last_result = None

    async def initialize(self) -> None:
        """
        Initialize the Drug Discovery Agent.

        This performs validation of the drug_discovery module availability.
        Called by BaseAgent.start().

        Raises:
            AgentInitializationError: If the drug_discovery module is not
                available or initialization fails.
        """
        if run_drug_discovery is None:
            raise AgentInitializationError(
                "Drug discovery module not available. "
                "Ensure drug_discovery.pipeline is installed and accessible.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info(
            "DrugDiscoveryAgent '%s' initialized with top_k=%d, similarity_threshold=%.2f",
            self.agent_id,
            self.default_top_k,
            self.default_similarity_threshold,
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

    def _extract_discovery_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract drug discovery parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of drug discovery parameters.
        """
        params = request.parameters or {}

        # Extract SMILES from parameters or instruction
        smiles = params.get("smiles", "")
        if not smiles and request.instruction:
            # Try to extract SMILES from instruction
            smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]+'
            matches = re.findall(smiles_pattern, request.instruction)
            if matches:
                # Use the longest match as the SMILES
                smiles = max(matches, key=len)

        # Extract compound name
        compound_name = params.get("compound_name", "")

        # Disease name for repurposing
        disease_name = params.get("disease_name", "")

        # Query SMILES for similarity
        query_smiles = params.get("query_smiles", smiles)

        # Configuration
        top_k = params.get("top_k", self.default_top_k)
        similarity_threshold = params.get(
            "similarity_threshold",
            self.default_similarity_threshold,
        )

        # Pipeline stages
        enable_cache = params.get("enable_cache", True)
        enable_similarity = params.get("enable_similarity", self.include_all_stages)
        enable_target_prediction = params.get("enable_target_prediction", self.include_all_stages)
        enable_toxicity_prediction = params.get("enable_toxicity_prediction", self.include_all_stages)
        enable_repurposing = params.get("enable_repurposing", self.include_all_stages)

        # Action type
        action = params.get("action", "full")

        return {
            "smiles": smiles,
            "compound_name": compound_name,
            "disease_name": disease_name,
            "query_smiles": query_smiles,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "enable_cache": enable_cache,
            "enable_similarity": enable_similarity,
            "enable_target_prediction": enable_target_prediction,
            "enable_toxicity_prediction": enable_toxicity_prediction,
            "enable_repurposing": enable_repurposing,
            "action": action,
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the drug discovery task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the drug discovery results.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_discovery_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        action = params["action"]

        if action == "properties":
            return await self._execute_properties(params, task)
        elif action == "similarity":
            return await self._execute_similarity(params, task)
        elif action == "targets":
            return await self._execute_targets(params, task)
        elif action == "toxicity":
            return await self._execute_toxicity(params, task)
        elif action == "repurposing":
            return await self._execute_repurposing(params, task)
        elif action == "ranking":
            return await self._execute_ranking(params, task)
        else:
            # Default: full pipeline
            return await self._execute_full_discovery(params, task)

    async def _execute_full_discovery(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a full drug discovery pipeline.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with drug discovery results.

        Raises:
            AgentExecutionError: If the pipeline fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for drug discovery analysis.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info("Executing full drug discovery analysis for: '%s'", smiles[:50])

        try:
            # Create compound
            compound_id = f"DISCOVERY_{task.request_id[:8]}"
            compound = Compound(
                compound_id=compound_id,
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            # Build configuration
            config = PipelineConfig(
                top_k=params["top_k"],
                enable_cache=params["enable_cache"],
                enable_similarity=params["enable_similarity"],
                enable_target_prediction=params["enable_target_prediction"],
                enable_toxicity_prediction=params["enable_toxicity_prediction"],
                enable_repurposing=params["enable_repurposing"],
            )
            config.discovery_config.similarity_threshold = params["similarity_threshold"]

            # Run the pipeline
            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=params["query_smiles"],
                disease_name=params.get("disease_name"),
                config=config,
            )

            # Store the result for follow-up questions
            self._last_result = result

            # Generate a summary
            summary = self._generate_summary(result, params)

            # Build output without type issues
            output: dict[str, Any] = {
                "success": True,
                "run_id": result.run_id,
                "stage": result.stage.value,
                "candidates": self._serialize_candidates(result.candidates),
                "rankings": self._serialize_rankings(result.rankings),
                "summary": summary,
                "candidate_count": len(result.candidates),
                "warnings": result.warnings,
                "metadata": result.metadata,
            }

            confidence = ConfidenceScore(
                value=0.85,
                basis=f"Drug discovery pipeline completed with {len(result.candidates)} candidates",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Drug discovery pipeline failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Drug discovery pipeline failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_properties(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a molecular properties analysis.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with molecular properties.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for property analysis.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            # Run full pipeline but only extract properties
            compound = Compound(
                compound_id=f"PROP_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=1,
                enable_cache=params["enable_cache"],
                enable_similarity=False,
                enable_target_prediction=False,
                enable_toxicity_prediction=False,
                enable_repurposing=False,
            )

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=smiles,
                disease_name=params.get("disease_name"),
                config=config,
            )

            self._last_result = result

            # Extract properties from the first candidate
            properties = None
            if result.candidates and result.candidates[0].molecular_properties:
                props = result.candidates[0].molecular_properties
                properties = self._serialize_properties(props)

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "properties": properties,
                "summary": self._generate_properties_summary(properties) if properties else "No properties available.",
            }

            confidence = ConfidenceScore(
                value=0.9,
                basis="Molecular properties computed successfully",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Property analysis failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Property analysis failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_similarity(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a similarity search.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with similarity results.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for similarity search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            # Run full pipeline but only extract similarity
            compound = Compound(
                compound_id=f"SIM_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=params["top_k"],
                enable_cache=params["enable_cache"],
                enable_similarity=True,
                enable_target_prediction=False,
                enable_toxicity_prediction=False,
                enable_repurposing=False,
            )
            config.discovery_config.similarity_threshold = params["similarity_threshold"]

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=params.get("query_smiles", smiles),
                disease_name=params.get("disease_name"),
                config=config,
            )

            self._last_result = result

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "similar_compounds": self._serialize_similarity(result),
                "similarity_count": len(result.candidates),
                "threshold": params["similarity_threshold"],
                "summary": f"Found {len(result.candidates)} similar compounds",
            }

            confidence = ConfidenceScore(
                value=0.8,
                basis=f"Similarity search found {len(result.candidates)} compounds",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Similarity search failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Similarity search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_targets(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute target prediction analysis.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with target predictions.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for target prediction.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            compound = Compound(
                compound_id=f"TARGET_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=1,
                enable_cache=params["enable_cache"],
                enable_similarity=False,
                enable_target_prediction=True,
                enable_toxicity_prediction=False,
                enable_repurposing=False,
            )

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=smiles,
                disease_name=params.get("disease_name"),
                config=config,
            )

            self._last_result = result

            # Extract targets from candidates
            targets = []
            for candidate in result.candidates:
                if candidate.target_predictions:
                    for pred in candidate.target_predictions:
                        targets.append({
                            "target": pred.target.name,
                            "type": pred.target.target_type.value,
                            "confidence": pred.confidence_score,
                            "confidence_level": pred.confidence_level.value,
                        })

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "targets": targets,
                "target_count": len(targets),
                "summary": f"Predicted {len(targets)} biological targets",
            }

            confidence = ConfidenceScore(
                value=0.75,
                basis=f"Target prediction found {len(targets)} targets",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Target prediction failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Target prediction failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_toxicity(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute toxicity prediction analysis.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with toxicity predictions.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for toxicity prediction.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            compound = Compound(
                compound_id=f"TOX_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=1,
                enable_cache=params["enable_cache"],
                enable_similarity=False,
                enable_target_prediction=False,
                enable_toxicity_prediction=True,
                enable_repurposing=False,
            )

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=smiles,
                disease_name=params.get("disease_name"),
                config=config,
            )

            self._last_result = result

            # Extract toxicity from candidates
            toxicity_info = None
            if result.candidates and result.candidates[0].toxicity_prediction:
                tox = result.candidates[0].toxicity_prediction
                toxicity_info = {
                    "level": tox.toxicity_level.value,
                    "score": tox.toxicity_score,
                    "confidence": tox.confidence.value,
                    "endpoints": tox.endpoint_scores,
                }

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "toxicity": toxicity_info,
                "summary": self._generate_toxicity_summary(toxicity_info) if toxicity_info else "No toxicity data available.",
            }

            confidence = ConfidenceScore(
                value=0.7,
                basis="Toxicity prediction completed",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Toxicity prediction failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Toxicity prediction failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_repurposing(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute drug repurposing analysis.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with repurposing recommendations.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for repurposing analysis.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        disease_name = params.get("disease_name", "")

        if not disease_name:
            self.logger.warning("No disease name provided for repurposing analysis")

        try:
            compound = Compound(
                compound_id=f"REPURPOSE_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=params["top_k"],
                enable_cache=params["enable_cache"],
                enable_similarity=False,
                enable_target_prediction=True,
                enable_toxicity_prediction=True,
                enable_repurposing=True,
            )

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=smiles,
                disease_name=disease_name or None,
                config=config,
            )

            self._last_result = result

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "disease_name": disease_name or "Not specified",
                "candidates": self._serialize_candidates(result.candidates),
                "candidate_count": len(result.candidates),
                "summary": f"Found {len(result.candidates)} potential repurposing candidates",
            }

            confidence = ConfidenceScore(
                value=0.75,
                basis=f"Drug repurposing analysis found {len(result.candidates)} candidates",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Drug repurposing failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Drug repurposing failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_ranking(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute candidate ranking analysis.

        Args:
            params: Request parameters.
            task: The task being executed.

        Returns:
            An AgentResult with ranking results.

        Raises:
            AgentExecutionError: If analysis fails.
        """
        smiles = params["smiles"]

        if not smiles:
            raise AgentExecutionError(
                "SMILES string is required for ranking analysis.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            compound = Compound(
                compound_id=f"RANK_{task.request_id[:8]}",
                name=params.get("compound_name", "Input Compound"),
                smiles=smiles,
            )

            config = PipelineConfig(
                top_k=params["top_k"],
                enable_cache=params["enable_cache"],
                enable_similarity=True,
                enable_target_prediction=True,
                enable_toxicity_prediction=True,
                enable_repurposing=True,
            )
            config.discovery_config.similarity_threshold = params["similarity_threshold"]

            result = run_drug_discovery(
                compounds=[compound],
                query_smiles=params.get("query_smiles", smiles),
                disease_name=params.get("disease_name"),
                config=config,
            )

            self._last_result = result

            output: dict[str, Any] = {
                "success": True,
                "smiles": smiles,
                "compound_name": params.get("compound_name", "Input Compound"),
                "rankings": self._serialize_rankings(result.rankings),
                "candidate_count": len(result.rankings),
                "summary": self._generate_ranking_summary(result.rankings),
            }

            confidence = ConfidenceScore(
                value=0.8,
                basis=f"Ranked {len(result.rankings)} candidates",
                source_module="drug_discovery",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Ranking analysis failed for '%s': %s", smiles, exc)
            raise AgentExecutionError(
                f"Ranking analysis failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    def _serialize_properties(self, props: Any) -> dict[str, Any]:
        """Serialize MolecularProperties to a dict."""
        if not props:
            return {}

        return {
            "molecular_weight": props.molecular_weight,
            "molecular_weight_formatted": f"{props.molecular_weight:.2f} g/mol",
            "logp": props.logp,
            "logp_formatted": f"{props.logp:.2f}",
            "tpsa": props.tpsa,
            "tpsa_formatted": f"{props.tpsa:.1f}",
            "h_bond_donors": props.h_bond_donors,
            "h_bond_acceptors": props.h_bond_acceptors,
            "rotatable_bonds": props.rotatable_bonds,
            "aromatic_rings": props.aromatic_rings,
            "heavy_atom_count": props.heavy_atom_count,
            "lipinski_violations": props.lipinski_violations,
            "qed_score": props.qed_score,
            "qed_formatted": f"{props.qed_score:.3f}" if props.qed_score else "N/A",
            "is_lipinski_compliant": props.is_lipinski_compliant() if hasattr(props, 'is_lipinski_compliant') else False,
        }

    def _serialize_candidates(self, candidates: list) -> list[dict[str, Any]]:
        """Serialize DrugCandidate list to dicts."""
        result = []
        for c in candidates[:10]:  # Limit to 10 for response size
            item: dict[str, Any] = {
                "candidate_id": c.candidate_id,
                "name": c.compound.name,
                "smiles": c.compound.smiles,
                "overall_score": c.overall_score,
                "score_formatted": f"{c.overall_score:.3f}" if c.overall_score else "N/A",
                "status": c.status.value if c.status else "pending",
            }

            if c.molecular_properties:
                item["properties"] = self._serialize_properties(c.molecular_properties)

            if c.target_predictions:
                item["target_count"] = len(c.target_predictions)
                item["top_target"] = c.target_predictions[0].target.name if c.target_predictions else None
                item["top_target_confidence"] = c.target_predictions[0].confidence_score if c.target_predictions else None

            if c.toxicity_prediction:
                item["toxicity_level"] = c.toxicity_prediction.toxicity_level.value
                item["toxicity_score"] = c.toxicity_prediction.toxicity_score

            result.append(item)
        return result

    def _serialize_rankings(self, rankings: list) -> list[dict[str, Any]]:
        """Serialize CandidateRanking list to dicts."""
        result = []
        for r in rankings[:10]:
            result.append({
                "rank": r.rank,
                "candidate_id": r.candidate_id,
                "composite_score": r.composite_score,
                "score_formatted": f"{r.composite_score:.3f}",
                "criteria_scores": r.criteria_scores,
                "notes": r.notes,
            })
        return result

    def _serialize_similarity(self, result: Any) -> list[dict[str, Any]]:
        """Extract similarity information from a DrugDiscoveryResult."""
        similar = []
        for candidate in result.candidates[:10]:
            # Look for similarity in metadata or use the candidate itself
            score = None
            if result.metadata and "similarity_scores" in result.metadata:
                score = result.metadata["similarity_scores"].get(candidate.compound.compound_id)

            similar.append({
                "name": candidate.compound.name,
                "smiles": candidate.compound.smiles,
                "similarity_score": score if score else 0.0,
                "score_formatted": f"{score:.3f}" if score else "N/A",
            })
        return similar

    def _generate_summary(self, result: Any, params: dict[str, Any]) -> str:
        """Generate a concise summary of the drug discovery result."""
        lines = [
            "Drug Discovery Analysis Summary",
            f"Run ID: {result.run_id}",
            f"Stage: {result.stage.value.replace('_', ' ').title()}",
            f"Candidates: {len(result.candidates)}",
            "",
        ]

        if result.candidates:
            lines.append("Top Candidates:")
            for i, candidate in enumerate(result.candidates[:3], 1):
                score = candidate.overall_score if candidate.overall_score else 0.0
                lines.append(f"  {i}. {candidate.compound.name} - Score: {score:.3f}")

            # Add molecular properties of top candidate
            top = result.candidates[0]
            if top.molecular_properties:
                props = top.molecular_properties
                lines.append("")
                lines.append(f"Top Candidate Properties ({top.compound.name}):")
                lines.append(f"  Molecular Weight: {props.molecular_weight:.2f} g/mol")
                lines.append(f"  LogP: {props.logp:.2f}")
                lines.append(f"  Lipinski Violations: {props.lipinski_violations}")
                if top.toxicity_prediction:
                    lines.append(f"  Toxicity Level: {top.toxicity_prediction.toxicity_level.value.upper()}")
                    lines.append(f"  Toxicity Score: {top.toxicity_prediction.toxicity_score:.3f}")

        if result.warnings:
            lines.append("")
            lines.append(f"Warnings: {len(result.warnings)}")

        return "\n".join(lines)

    def _generate_properties_summary(self, props: Any) -> str:
        """Generate a human-readable summary of molecular properties."""
        if not props:
            return "No molecular properties available."

        lines = [
            f"Molecular Weight: {props.get('molecular_weight', 0):.2f} g/mol",
            f"LogP: {props.get('logp', 0):.2f}",
            f"TPSA: {props.get('tpsa', 0):.1f}",
            f"H-Bond Donors: {props.get('h_bond_donors', 0)}",
            f"H-Bond Acceptors: {props.get('h_bond_acceptors', 0)}",
            f"Rotatable Bonds: {props.get('rotatable_bonds', 0)}",
            f"Lipinski Violations: {props.get('lipinski_violations', 0)}",
            f"QED Score: {props.get('qed_formatted', 'N/A')}",
            f"Lipinski Compliant: {'Yes' if props.get('is_lipinski_compliant', False) else 'No'}",
        ]
        return "\n".join(lines)

    def _generate_toxicity_summary(self, toxicity_info: Optional[dict]) -> str:
        """Generate a human-readable summary of toxicity predictions."""
        if not toxicity_info:
            return "No toxicity data available."

        lines = [
            f"Toxicity Level: {toxicity_info.get('level', 'unknown').upper()}",
            f"Toxicity Score: {toxicity_info.get('score', 0):.3f}",
            f"Confidence: {toxicity_info.get('confidence', 'unknown').upper()}",
        ]

        endpoints = toxicity_info.get('endpoints')
        if endpoints:
            lines.append("Endpoint Scores:")
            for endpoint, score in endpoints.items():
                lines.append(f"  {endpoint}: {score:.3f}")

        return "\n".join(lines)

    def _generate_ranking_summary(self, rankings: list) -> str:
        """Generate a human-readable summary of candidate rankings."""
        if not rankings:
            return "No rankings available."

        lines = ["Candidate Rankings:"]
        for r in rankings[:5]:
            lines.append(f"  {r.rank}. Score: {r.composite_score:.3f}")
            if r.criteria_scores:
                top_criteria = sorted(r.criteria_scores.items(), key=lambda x: -x[1])[:3]
                criteria_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_criteria)
                lines.append(f"     Criteria: {criteria_str}")

        return "\n".join(lines)

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        This releases any resources acquired during initialization.
        """
        self.logger.info("Shutting down DrugDiscoveryAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "drug_discovery",
            "molecular_properties",
            "compound_similarity",
            "target_prediction",
            "toxicity_prediction",
            "drug_repurposing",
            "candidate_ranking",
            "compound_analysis",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"DrugDiscoveryAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_drug_discovery_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    default_top_k: int = 25,
    default_similarity_threshold: float = 0.7,
    include_all_stages: bool = True,
) -> DrugDiscoveryAgent:
    """
    Factory function to create a DrugDiscoveryAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        default_top_k: Default number of candidates to return.
        default_similarity_threshold: Default similarity threshold.
        include_all_stages: Whether to run all pipeline stages.

    Returns:
        A configured DrugDiscoveryAgent instance.
    """
    return DrugDiscoveryAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        default_top_k=default_top_k,
        default_similarity_threshold=default_similarity_threshold,
        include_all_stages=include_all_stages,
    )


__all__: list[str] = [
    "DrugDiscoveryAgent",
    "create_drug_discovery_agent",
]