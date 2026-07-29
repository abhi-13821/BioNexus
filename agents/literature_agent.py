"""
agents/literature_agent.py

Literature Agent for the BioNexus Multi-Agent AI System.

This agent specializes in biomedical literature retrieval, search, and
analysis. It wraps the existing literature_search module (frontend/literature_search.py)
and exposes its capabilities through the standardized BaseAgent interface.

The agent is responsible for:
    - Handling literature-related queries (search, summarization, analysis)
    - Using the federated search across PubMed, Europe PMC, CrossRef, etc.
    - Summarizing search results with AI or heuristic methods
    - Ranking and filtering papers based on relevance
    - Answering literature-specific questions
    - Maintaining conversation context through Memory
    - Returning structured responses compatible with agents.models

Design decisions:
    - The agent does NOT reimplement literature search logic; it delegates
      to the existing literature_search module's functions.
    - All heavy lifting (federated search, deduplication, ranking, export,
      visualization) is done by the literature_search module.
    - This agent only orchestrates: validates requests, calls the module,
      formats results, and manages memory.

Integration Points:
    - frontend.literature_search: run_federated_search, Paper, SearchFilters,
      generate_summaries, generate_literature_review, citation_summary
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
from datetime import datetime, timezone
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
    ExecutionMetadata,
    TaskStatus,
)
from drug_discovery.utils import generate_unique_id

# Import literature search functionality
try:
    from frontend.literature_search import (
        Paper,
        SearchFilters,
        citation_summary,
        generate_literature_review,
        generate_summaries,
        run_federated_search,
        Config as LiteratureConfig,
    )
except ImportError:
    # Fallback for testing/standalone usage
    Paper = Any
    SearchFilters = Any
    run_federated_search = None
    generate_summaries = None
    generate_literature_review = None
    citation_summary = None
    LiteratureConfig = None

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# LiteratureAgent
# ----------------------------------------------------------------------


class LiteratureAgent(BaseAgent):
    """
    Agent specialized in biomedical literature retrieval and analysis.

    This agent wraps the existing literature_search module and provides
    a standardized interface for the multi-agent system. It handles
    literature search, summarization, citation analysis, and review
    generation.

    Example:
        >>> agent = LiteratureAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Find papers about EGFR mutations in lung cancer",
        ...     parameters={"query": "EGFR mutation lung cancer", "max_results": 20}
        ... )
        >>> response = await agent.handle_request(request)
        >>> print(response.message)

    Attributes:
        memory: MemoryManager instance for conversation history.
        default_max_results: Default number of papers to return.
        default_sources: Default list of sources to search.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        default_max_results: int = 40,
        default_sources: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the Literature Agent.

        Args:
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default AgentConfig is used if not supplied.
            memory: MemoryManager instance for conversation history.
                A new InMemoryMemoryStore is used if not supplied.
            default_max_results: Default number of papers to return per
                search. Defaults to 40.
            default_sources: Default list of sources to search. Defaults
                to all sources defined in LiteratureConfig.SOURCES.
        """
        super().__init__(
            agent_type=AgentType.LITERATURE_SEARCH_AGENT,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=120.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=False,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.default_max_results = default_max_results
        self.default_sources = default_sources or (
            LiteratureConfig.SOURCES if LiteratureConfig else [
                "PubMed", "PubMed Central", "Europe PMC", "CrossRef",
                "Semantic Scholar", "OpenAlex", "DOAJ", "bioRxiv", "medRxiv",
            ]
        )
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the Literature Agent.

        This performs validation of the literature_search module availability
        and warms up any required resources. Called by BaseAgent.start().

        Raises:
            AgentInitializationError: If the literature_search module is
                not available or initialization fails.
        """
        if run_federated_search is None:
            raise AgentInitializationError(
                "Literature search module not available. "
                "Ensure frontend.literature_search is installed and accessible.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info(
            "LiteratureAgent '%s' initialized with %d default sources.",
            self.agent_id,
            len(self.default_sources),
        )

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

        if self._initialized is False:
            raise AgentNotInitializedError(
                f"Agent '{self.agent_id}' ({self.agent_type.value}) "
                "received a request before start() was called.",
                agent_type=self.agent_type,
                is_recoverable=True,
            )

    def _extract_search_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract search parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of search parameters.
        """
        params = request.parameters or {}

        # Extract query from instruction or parameters
        query = params.get("query", "")
        if not query and request.instruction:
            # Use instruction as query if no explicit query parameter
            query = request.instruction

        # Build filters
        filters = SearchFilters() if SearchFilters else None

        if filters and params.get("filters"):
            filter_dict = params["filters"]
            for key, value in filter_dict.items():
                if hasattr(filters, key):
                    setattr(filters, key, value)

        # Override max_results if specified
        max_results = params.get("max_results", self.default_max_results)
        if filters:
            filters.max_results = max_results

        # Sources
        sources = params.get("sources", self.default_sources)

        # Action type: search, summarize, review, analyze
        action = params.get("action", "search")

        return {
            "query": query,
            "filters": filters,
            "sources": sources,
            "max_results": max_results,
            "action": action,
            "paper_ids": params.get("paper_ids", []),
            "top_n": params.get("top_n", 10),
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the literature search task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the literature search results.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_search_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        action = params["action"]
        query = params["query"]

        if not query:
            raise AgentExecutionError(
                "Query is required for literature search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            if action == "search":
                result_data = await self._execute_search(params)
            elif action == "summarize":
                result_data = await self._execute_summarize(params)
            elif action == "review":
                result_data = await self._execute_review(params)
            elif action == "analyze":
                result_data = await self._execute_analyze(params)
            else:
                # Default to search
                result_data = await self._execute_search(params)

            # Record the result in memory
            confidence = ConfidenceScore(
                value=0.85,
                basis=f"Literature search completed with {result_data.get('paper_count', 0)} papers",
                source_module="literature_search",
            )

            return self._build_success_result(
                task=task,
                output=result_data,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Literature search execution failed: %s", exc)
            raise AgentExecutionError(
                f"Literature search execution failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a literature search.

        Args:
            params: Search parameters.

        Returns:
            A dictionary containing search results and metadata.
        """
        query = params["query"]
        filters = params["filters"]
        sources = params["sources"]

        self.logger.info(
            "Executing literature search: query='%s', sources=%d, max_results=%d",
            query[:50],
            len(sources),
            filters.max_results if filters else self.default_max_results,
        )

        # Run the federated search
        papers, status = run_federated_search(
            query=query,
            filters=filters or SearchFilters(),
            sources=sources,
        )

        # Generate summaries for papers
        for paper in papers[:20]:
            if not paper.short_summary:
                generate_summaries(paper)

        # Prepare results
        result_dicts = []
        for paper in papers[:50]:
            paper_dict = asdict(paper)
            # Convert any non-serializable types
            if paper_dict.get("entities"):
                paper_dict["entities"] = {
                    k: list(v) for k, v in paper_dict["entities"].items()
                }
            result_dicts.append(paper_dict)

        # Calculate citation summary
        citation_stats = citation_summary(papers) if citation_summary else {}

        return {
            "action": "search",
            "query": query,
            "paper_count": len(papers),
            "papers": result_dicts,
            "source_status": status,
            "citation_summary": citation_stats,
        }

    async def _execute_summarize(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a summarization request.

        Args:
            params: Request parameters.

        Returns:
            A dictionary containing summaries.
        """
        # If we have paper IDs, fetch those papers from results
        # Otherwise, search first then summarize
        if params.get("paper_ids"):
            # For now, search and filter by IDs
            # In a real implementation, this would fetch from memory/store
            search_result = await self._execute_search(params)
            papers = search_result.get("papers", [])
            # Filter by IDs if provided
            paper_ids = set(params["paper_ids"])
            filtered = [p for p in papers if p.get("uid") in paper_ids]
            search_result["papers"] = filtered
            search_result["paper_count"] = len(filtered)
            search_result["action"] = "summarize"
            return search_result

        # Otherwise search and summarize all
        return await self._execute_search(params)

    async def _execute_review(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a literature review generation request.

        Args:
            params: Request parameters.

        Returns:
            A dictionary containing the generated review.
        """
        query = params["query"]

        # Search first
        search_result = await self._execute_search(params)
        papers_data = search_result.get("papers", [])

        if not papers_data:
            return {
                "action": "review",
                "query": query,
                "review": "No papers found to generate a review.",
                "paper_count": 0,
            }

        # Reconstruct Paper objects from dicts
        papers = []
        for p_data in papers_data[:50]:
            # Create a Paper object from the dict
            # This assumes Paper can be constructed from the dict
            paper = Paper(**p_data) if Paper else None
            if paper:
                papers.append(paper)

        # Generate review
        review_text = generate_literature_review(papers, query) if generate_literature_review else ""

        return {
            "action": "review",
            "query": query,
            "review": review_text,
            "paper_count": len(papers),
        }

    async def _execute_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a citation/analysis request.

        Args:
            params: Request parameters.

        Returns:
            A dictionary containing analysis results.
        """
        # Search first
        search_result = await self._execute_search(params)
        papers_data = search_result.get("papers", [])
        citation_stats = search_result.get("citation_summary", {})

        # Additional analysis could go here

        return {
            "action": "analyze",
            "query": params["query"],
            "paper_count": len(papers_data),
            "citation_summary": citation_stats,
            "analysis": {
                "total_papers": len(papers_data),
                "top_cited": citation_stats.get("top_cited", []),
                "year_range": citation_stats.get("year_range", {}),
            },
        }

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        This releases any resources acquired during initialization.
        """
        self.logger.info("Shutting down LiteratureAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "literature_search",
            "literature_summarize",
            "literature_review",
            "literature_analyze",
            "citation_analysis",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"LiteratureAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_literature_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    default_max_results: int = 40,
    default_sources: Optional[list[str]] = None,
) -> LiteratureAgent:
    """
    Factory function to create a LiteratureAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        default_max_results: Default number of papers to return per search.
        default_sources: Default list of sources to search.

    Returns:
        A configured LiteratureAgent instance.
    """
    return LiteratureAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        default_max_results=default_max_results,
        default_sources=default_sources,
    )


__all__: list[str] = [
    "LiteratureAgent",
    "create_literature_agent",
]