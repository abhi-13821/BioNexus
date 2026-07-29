"""
agents/drug_agent.py

Drug Agent for the BioNexus Multi-Agent AI System.

This agent specializes in drug information retrieval, analysis, and
recommendation. It wraps the existing drug_info module (frontend/drug_info.py)
and exposes its capabilities through the standardized BaseAgent interface.

The agent is responsible for:
    - Handling drug-related queries (search, properties, pharmacology)
    - Using the drug_info module for PubChem data retrieval
    - Answering questions about drug properties, mechanisms, indications
    - Recommending similar drugs based on structure or mechanism
    - Maintaining conversation context through Memory
    - Returning structured responses compatible with agents.models

Design decisions:
    - The agent does NOT reimplement drug info logic; it delegates to
      the existing drug_info module's functions.
    - All heavy lifting (PubChem API calls, property parsing, embeddings)
      is done by the drug_info module.
    - This agent only orchestrates: validates requests, calls the module,
      formats results, and manages memory.

Integration Points:
    - frontend.drug_info: fetch_compound_properties, CompoundProperties,
      evaluate_lipinski, find_similar_drugs, find_similar_by_mechanism,
      find_similar_by_indication, semantic_drug_search, ingest_drug_embeddings
    - agents.base_agent: BaseAgent
    - agents.models: AgentRequest, AgentTask, AgentResult, AgentResponse
    - agents.memory: MemoryManager

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
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

# Import drug_info functionality with fallback
try:
    from frontend.drug_info import (  # type: ignore
        CompoundProperties,
        evaluate_lipinski,
        fetch_compound_properties,
        find_similar_drugs,
        find_similar_by_indication,
        find_similar_by_mechanism,
        ingest_drug_embeddings,
        semantic_drug_search,
    )
except ImportError:
    # Fallback for testing/standalone usage
    CompoundProperties = Any  # type: ignore
    evaluate_lipinski = None  # type: ignore
    fetch_compound_properties = None  # type: ignore
    find_similar_drugs = None  # type: ignore
    find_similar_by_indication = None  # type: ignore
    find_similar_by_mechanism = None  # type: ignore
    ingest_drug_embeddings = None  # type: ignore
    semantic_drug_search = None  # type: ignore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DrugAgent
# ----------------------------------------------------------------------


class DrugAgent(BaseAgent):
    """
    Agent specialized in drug information retrieval and analysis.

    This agent wraps the existing drug_info module and provides a
    standardized interface for the multi-agent system. It handles drug
    search, property analysis, similarity recommendations, and
    pharmacological information.

    Example:
        >>> agent = DrugAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Tell me about Aspirin",
        ...     parameters={"drug_name": "Aspirin"}
        ... )
        >>> response = await agent.handle_request(request)
        >>> print(response.message)

    Attributes:
        memory: MemoryManager instance for conversation history.
        default_top_k: Default number of similar drugs to return.
        enable_embeddings: Whether to use semantic embeddings for recommendations.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        default_top_k: int = 5,
        enable_embeddings: bool = True,
    ) -> None:
        """
        Initialize the Drug Agent.

        Args:
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default AgentConfig is used if not supplied.
            memory: MemoryManager instance for conversation history.
                A new InMemoryMemoryStore is used if not supplied.
            default_top_k: Default number of similar drug recommendations
                to return. Defaults to 5.
            enable_embeddings: Whether to enable semantic embedding-based
                similarity search. Defaults to True.
        """
        super().__init__(
            agent_type=AgentType.DRUG_INFO_AGENT,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=60.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=False,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.default_top_k = default_top_k
        self.enable_embeddings = enable_embeddings
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the Drug Agent.

        This performs validation of the drug_info module availability
        and warms up any required resources. Called by BaseAgent.start().

        Raises:
            AgentInitializationError: If the drug_info module is not
                available or initialization fails.
        """
        if fetch_compound_properties is None:
            raise AgentInitializationError(
                "Drug info module not available. "
                "Ensure frontend.drug_info is installed and accessible.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info(
            "DrugAgent '%s' initialized with default_top_k=%d, enable_embeddings=%s",
            self.agent_id,
            self.default_top_k,
            self.enable_embeddings,
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

    def _extract_drug_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract drug search parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of drug search parameters.
        """
        params = request.parameters or {}

        # Extract drug name from parameters or instruction
        drug_name = params.get("drug_name", "")
        if not drug_name and request.instruction:
            # Try to extract drug name from instruction
            drug_name = request.instruction

        # Action type: search, properties, similar, mechanism, indication, semantic
        action = params.get("action", "search")

        # Top K for similarity searches
        top_k = params.get("top_k", self.default_top_k)

        # Whether to include embeddings-based recommendations
        include_embeddings = params.get("include_embeddings", self.enable_embeddings)

        return {
            "drug_name": drug_name,
            "action": action,
            "top_k": top_k,
            "include_embeddings": include_embeddings,
            "query": params.get("query", ""),  # For semantic search
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the drug information task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the drug information results.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_drug_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        action = params["action"]

        if action == "semantic":
            return await self._execute_semantic_search(params, task)
        elif action == "similar":
            return await self._execute_similar_drugs(params, task)
        elif action == "mechanism":
            return await self._execute_similar_mechanism(params, task)
        elif action == "indication":
            return await self._execute_similar_indication(params, task)
        else:
            return await self._execute_drug_search(params, task)

    async def _execute_drug_search(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a full drug search and return comprehensive information.

        Args:
            params: Search parameters.
            task: The task being executed.

        Returns:
            An AgentResult with drug information.

        Raises:
            AgentExecutionError: If the search fails.
        """
        drug_name = params["drug_name"]

        if not drug_name:
            raise AgentExecutionError(
                "Drug name is required for search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info("Executing drug search: '%s'", drug_name)

        try:
            props = fetch_compound_properties(drug_name)

            if props is None:
                raise AgentExecutionError(
                    f"Drug '{drug_name}' not found in PubChem. "
                    "Please check the spelling or try a different drug name.",
                    agent_type=self.agent_type,
                    is_recoverable=True,
                )

            # Ingest embeddings for similarity search
            if self.enable_embeddings and ingest_drug_embeddings:
                try:
                    ingest_stats = ingest_drug_embeddings(props, drug_name)
                    self.logger.debug(
                        "Embedding ingestion stats for '%s': %s",
                        drug_name,
                        ingest_stats,
                    )
                except Exception as e:
                    self.logger.warning("Failed to ingest embeddings for '%s': %s", drug_name, e)

            # Evaluate Lipinski
            lipinski = None
            if evaluate_lipinski:
                try:
                    lipinski = evaluate_lipinski(props)
                except Exception as e:
                    self.logger.warning("Failed to evaluate Lipinski for '%s': %s", drug_name, e)

            # Build output
            output = {
                "drug_name": drug_name,
                "compound": asdict(props),
                "lipinski": lipinski,
                "found": True,
            }

            # Add similar drugs if available
            if self.enable_embeddings and find_similar_drugs:
                try:
                    similar = find_similar_drugs(props, top_k=params["top_k"])
                    output["similar_drugs"] = [
                        {
                            "name": self._extract_drug_name_from_result(r),
                            "score": r.score,
                            "cid": self._extract_cid_from_result(r),
                        }
                        for r in similar[:params["top_k"]]
                    ]
                except Exception as e:
                    self.logger.warning("Failed to find similar drugs for '%s': %s", drug_name, e)
                    output["similar_drugs"] = []

            confidence = ConfidenceScore(
                value=0.9,
                basis=f"Drug '{drug_name}' found in PubChem with CID {props.cid}",
                source_module="drug_info",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Drug search execution failed for '%s': %s", drug_name, exc)
            raise AgentExecutionError(
                f"Drug search execution failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_similar_drugs(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a similar drugs search.

        Args:
            params: Search parameters.
            task: The task being executed.

        Returns:
            An AgentResult with similar drug recommendations.

        Raises:
            AgentExecutionError: If the search fails.
        """
        drug_name = params["drug_name"]

        if not drug_name:
            raise AgentExecutionError(
                "Drug name is required for similar drug search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            props = fetch_compound_properties(drug_name)

            if props is None:
                raise AgentExecutionError(
                    f"Drug '{drug_name}' not found in PubChem.",
                    agent_type=self.agent_type,
                    is_recoverable=True,
                )

            if not find_similar_drugs:
                raise AgentExecutionError(
                    "Similar drug search is not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            # Ingest embeddings first
            if ingest_drug_embeddings:
                try:
                    ingest_drug_embeddings(props, drug_name)
                except Exception:
                    pass

            similar = find_similar_drugs(props, top_k=params["top_k"])

            output = {
                "drug_name": drug_name,
                "similar_drugs": [
                    {
                        "name": self._extract_drug_name_from_result(r),
                        "score": r.score,
                        "cid": self._extract_cid_from_result(r),
                    }
                    for r in similar
                ],
                "top_k": params["top_k"],
            }

            confidence = ConfidenceScore(
                value=0.8,
                basis=f"Found {len(similar)} similar drugs to '{drug_name}'",
                source_module="drug_info",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Similar drugs search failed for '%s': %s", drug_name, exc)
            raise AgentExecutionError(
                f"Similar drugs search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_similar_mechanism(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a similar mechanism of action search.

        Args:
            params: Search parameters.
            task: The task being executed.

        Returns:
            An AgentResult with mechanism-similar drug recommendations.

        Raises:
            AgentExecutionError: If the search fails.
        """
        drug_name = params["drug_name"]

        if not drug_name:
            raise AgentExecutionError(
                "Drug name is required for mechanism search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            props = fetch_compound_properties(drug_name)

            if props is None:
                raise AgentExecutionError(
                    f"Drug '{drug_name}' not found in PubChem.",
                    agent_type=self.agent_type,
                    is_recoverable=True,
                )

            if not find_similar_by_mechanism:
                raise AgentExecutionError(
                    "Mechanism-based similarity search is not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            # Ingest embeddings first
            if ingest_drug_embeddings:
                try:
                    ingest_drug_embeddings(props, drug_name)
                except Exception:
                    pass

            similar = find_similar_by_mechanism(props, top_k=params["top_k"])

            output = {
                "drug_name": drug_name,
                "similar_by_mechanism": [
                    {
                        "name": self._extract_drug_name_from_result(r),
                        "score": r.score,
                        "cid": self._extract_cid_from_result(r),
                    }
                    for r in similar
                ],
                "top_k": params["top_k"],
            }

            confidence = ConfidenceScore(
                value=0.75,
                basis=f"Found {len(similar)} drugs with similar mechanism to '{drug_name}'",
                source_module="drug_info",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Mechanism similarity search failed for '%s': %s", drug_name, exc)
            raise AgentExecutionError(
                f"Mechanism similarity search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_similar_indication(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a similar indication search.

        Args:
            params: Search parameters.
            task: The task being executed.

        Returns:
            An AgentResult with indication-similar drug recommendations.

        Raises:
            AgentExecutionError: If the search fails.
        """
        drug_name = params["drug_name"]

        if not drug_name:
            raise AgentExecutionError(
                "Drug name is required for indication search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            props = fetch_compound_properties(drug_name)

            if props is None:
                raise AgentExecutionError(
                    f"Drug '{drug_name}' not found in PubChem.",
                    agent_type=self.agent_type,
                    is_recoverable=True,
                )

            if not find_similar_by_indication:
                raise AgentExecutionError(
                    "Indication-based similarity search is not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            # Ingest embeddings first
            if ingest_drug_embeddings:
                try:
                    ingest_drug_embeddings(props, drug_name)
                except Exception:
                    pass

            similar = find_similar_by_indication(props, top_k=params["top_k"])

            output = {
                "drug_name": drug_name,
                "similar_by_indication": [
                    {
                        "name": self._extract_drug_name_from_result(r),
                        "score": r.score,
                        "cid": self._extract_cid_from_result(r),
                    }
                    for r in similar
                ],
                "top_k": params["top_k"],
            }

            confidence = ConfidenceScore(
                value=0.75,
                basis=f"Found {len(similar)} drugs with similar indications to '{drug_name}'",
                source_module="drug_info",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Indication similarity search failed for '%s': %s", drug_name, exc)
            raise AgentExecutionError(
                f"Indication similarity search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_semantic_search(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a semantic drug search.

        Args:
            params: Search parameters.
            task: The task being executed.

        Returns:
            An AgentResult with semantically matching drugs.

        Raises:
            AgentExecutionError: If the search fails.
        """
        query = params.get("query", "")

        if not query or not query.strip():
            raise AgentExecutionError(
                "Query text is required for semantic search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        if not semantic_drug_search:
            raise AgentExecutionError(
                "Semantic drug search is not available.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            results = semantic_drug_search(query, top_k=params["top_k"])

            output = {
                "query": query,
                "results": [
                    {
                        "name": self._extract_drug_name_from_result(r),
                        "score": r.score,
                        "cid": self._extract_cid_from_result(r),
                    }
                    for r in results
                ],
                "top_k": params["top_k"],
            }

            confidence = ConfidenceScore(
                value=0.7,
                basis=f"Semantic search for '{query}' returned {len(results)} results",
                source_module="drug_info",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Semantic drug search failed for '%s': %s", query, exc)
            raise AgentExecutionError(
                f"Semantic drug search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    def _extract_drug_name_from_result(self, result: Any) -> str:
        """
        Extract the drug name from a similarity search result.

        Args:
            result: A SimilaritySearchResult object.

        Returns:
            The drug name as a string.
        """
        try:
            # Try to get from metadata extra
            if hasattr(result, 'record') and hasattr(result.record, 'metadata'):
                extra = getattr(result.record.metadata, 'extra', {})
                if extra and 'drug_name' in extra:
                    return str(extra['drug_name'])
                if hasattr(result.record.metadata, 'title'):
                    return str(result.record.metadata.title)
            # Fallback: use the record ID
            if hasattr(result, 'record') and hasattr(result.record, 'source_id'):
                return f"Drug_{result.record.source_id}"
            return "Unknown Drug"
        except Exception:
            return "Unknown Drug"

    def _extract_cid_from_result(self, result: Any) -> Optional[int]:
        """
        Extract the PubChem CID from a similarity search result.

        Args:
            result: A SimilaritySearchResult object.

        Returns:
            The PubChem CID as an integer, or None if not found.
        """
        try:
            if hasattr(result, 'record') and hasattr(result.record, 'metadata'):
                extra = getattr(result.record.metadata, 'extra', {})
                if extra and 'cid' in extra:
                    cid = extra['cid']
                    if isinstance(cid, int):
                        return cid
                    if isinstance(cid, str) and cid.isdigit():
                        return int(cid)
                # Try to get from source_id
                if hasattr(result.record, 'source_id'):
                    source_id = result.record.source_id
                    if isinstance(source_id, str) and source_id.isdigit():
                        return int(source_id)
                    if isinstance(source_id, int):
                        return source_id
            return None
        except Exception:
            return None

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        This releases any resources acquired during initialization.
        """
        self.logger.info("Shutting down DrugAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "drug_search",
            "drug_properties",
            "drug_similarity",
            "drug_mechanism",
            "drug_indication",
            "drug_lipinski",
            "drug_semantic_search",
            "drug_synonyms",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"DrugAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_drug_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    default_top_k: int = 5,
    enable_embeddings: bool = True,
) -> DrugAgent:
    """
    Factory function to create a DrugAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        default_top_k: Default number of similar drug recommendations to return.
        enable_embeddings: Whether to enable semantic embedding-based similarity.

    Returns:
        A configured DrugAgent instance.
    """
    return DrugAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        default_top_k=default_top_k,
        enable_embeddings=enable_embeddings,
    )


__all__: list[str] = [
    "DrugAgent",
    "create_drug_agent",
]