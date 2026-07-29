"""
agents/knowledge_graph_agent.py

Knowledge Graph Agent for the BioNexus Multi-Agent AI System.

This agent specializes in biomedical knowledge graph exploration and
reasoning. It wraps the existing knowledge_graphs module (frontend/knowledge_graphs.py)
and exposes its capabilities through the standardized BaseAgent interface.

The agent is responsible for:
    - Handling biomedical entity relationship queries
    - Exploring connections between genes, diseases, proteins, drugs, pathways
    - Supporting graph exploration and relationship reasoning
    - Explaining graph results in natural language
    - Maintaining conversation context through Memory
    - Returning structured responses compatible with agents.models
    - Handling missing entities gracefully
    - Supporting follow-up graph queries

Design decisions:
    - The agent does NOT reimplement graph logic; it delegates to
      the existing knowledge_graphs module's functions.
    - All heavy lifting (entity extraction, graph building, filtering,
      embedding sync, semantic search) is done by the knowledge_graphs module.
    - This agent only orchestrates: validates requests, calls the module,
      formats results, and manages memory.

Integration Points:
    - frontend.knowledge_graphs: KnowledgeGraphPipeline, KGConfig,
      _build_filtered_graph, _render_graph, _get_embedding_manager,
      _sync_entity_embeddings, _sync_relation_embeddings,
      _compute_embedding_layout, _cluster_entities_by_embedding
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

# Import knowledge_graphs functionality with fallback
try:
    from frontend.knowledge_graphs import (
        KGConfig,
        _build_filtered_graph,
        _get_embedding_manager,
        _sync_entity_embeddings,
        _sync_relation_embeddings,
        _compute_embedding_layout,
        _cluster_entities_by_embedding,
    )
    from knowledge_graph.kg_pipeline import KnowledgeGraphPipeline
except ImportError:
    # Fallback for testing/standalone usage
    KGConfig = Any  # type: ignore
    _build_filtered_graph = None  # type: ignore
    _get_embedding_manager = None  # type: ignore
    _sync_entity_embeddings = None  # type: ignore
    _sync_relation_embeddings = None  # type: ignore
    _compute_embedding_layout = None  # type: ignore
    _cluster_entities_by_embedding = None  # type: ignore
    KnowledgeGraphPipeline = None  # type: ignore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# KnowledgeGraphAgent
# ----------------------------------------------------------------------


class KnowledgeGraphAgent(BaseAgent):
    """
    Agent specialized in biomedical knowledge graph exploration and reasoning.

    This agent wraps the existing knowledge_graphs module and provides a
    standardized interface for the multi-agent system. It handles graph
    construction, entity relationship queries, semantic search, clustering,
    and graph expansion.

    Example:
        >>> agent = KnowledgeGraphAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Find relationships between BRCA1 and breast cancer",
        ...     parameters={"entity1": "BRCA1", "entity2": "breast cancer"}
        ... )
        >>> response = await agent.handle_request(request)
        >>> print(response.message)

    Attributes:
        memory: MemoryManager instance for conversation history.
        default_max_nodes: Default maximum nodes to display.
        default_max_edges: Default maximum edges to display.
        min_importance: Minimum importance threshold for nodes.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        default_max_nodes: int = 30,
        default_max_edges: int = 50,
        min_importance: float = 0.0,
    ) -> None:
        """
        Initialize the Knowledge Graph Agent.

        Args:
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default AgentConfig is used if not supplied.
            memory: MemoryManager instance for conversation history.
                A new InMemoryMemoryStore is used if not supplied.
            default_max_nodes: Default maximum number of nodes to display.
                Defaults to 30.
            default_max_edges: Default maximum number of edges to display.
                Defaults to 50.
            min_importance: Minimum importance threshold for nodes.
                Defaults to 0.0.
        """
        super().__init__(
            agent_type=AgentType.KNOWLEDGE_GRAPH_AGENT,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=120.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=True,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.default_max_nodes = default_max_nodes
        self.default_max_edges = default_max_edges
        self.min_importance = min_importance
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the Knowledge Graph Agent.

        This performs validation of the knowledge_graphs module availability
        and warms up any required resources. Called by BaseAgent.start().

        Raises:
            AgentInitializationError: If the knowledge_graphs module is not
                available or initialization fails.
        """
        if KnowledgeGraphPipeline is None:
            raise AgentInitializationError(
                "Knowledge Graph pipeline not available. "
                "Ensure knowledge_graph.kg_pipeline is installed and accessible.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        self.logger.info(
            "KnowledgeGraphAgent '%s' initialized with max_nodes=%d, max_edges=%d",
            self.agent_id,
            self.default_max_nodes,
            self.default_max_edges,
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

    def _extract_graph_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract graph query parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of graph query parameters.
        """
        params = request.parameters or {}

        # Extract entity from parameters or instruction
        entity1 = params.get("entity1", "")
        entity2 = params.get("entity2", "")

        # If no entity1, try to extract from instruction
        if not entity1 and request.instruction:
            # Try to find entity mentions in instruction
            import re
            # Look for common patterns like "X and Y" or "X related to Y"
            pattern = r'(?:between|among|of|for|with|and)\s+([A-Za-z0-9\s\-_]+?)(?:\s+and\s+|\s+related to\s+|\s+with\s+)([A-Za-z0-9\s\-_]+)'
            match = re.search(pattern, request.instruction, re.IGNORECASE)
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()

        # Action type: query, explore, neighbors, path, semantic, cluster
        action = params.get("action", "query")

        # Graph parameters
        max_nodes = params.get("max_nodes", self.default_max_nodes)
        max_edges = params.get("max_edges", self.default_max_edges)
        min_importance = params.get("min_importance", self.min_importance)

        # Semantic search parameters
        semantic_query = params.get("semantic_query", "")

        return {
            "entity1": entity1,
            "entity2": entity2,
            "action": action,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "min_importance": min_importance,
            "semantic_query": semantic_query,
            "top_k": params.get("top_k", 10),
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the knowledge graph task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the knowledge graph results.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_graph_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        action = params["action"]

        if action == "semantic":
            return await self._execute_semantic_search(params, task)
        elif action == "cluster":
            return await self._execute_cluster(params, task)
        elif action == "neighbors":
            return await self._execute_neighbors(params, task)
        else:
            # Default: query or explore
            return await self._execute_graph_query(params, task)

    async def _execute_graph_query(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a graph query to find relationships between entities.

        Args:
            params: Query parameters.
            task: The task being executed.

        Returns:
            An AgentResult with graph query results.

        Raises:
            AgentExecutionError: If the query fails.
        """
        # This requires papers to be available in session state
        # We need to get literature search results first
        # For now, we'll check if results exist

        try:
            # Try to get literature results from session state
            # Since this is an agent, we need to access this differently
            # For now, we'll return a message about needing literature search first
            import streamlit as st

            results = st.session_state.get("bx_last_results", [])

            if not results:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No literature search results available. "
                        "Please run a literature search first to build a knowledge graph.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            # Build the knowledge graph
            pipeline = KnowledgeGraphPipeline()
            process_result = pipeline.process_papers(results)
            raw_graph = process_result.get("graph")

            if raw_graph is None or raw_graph.number_of_nodes() == 0:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No entities or relationships could be extracted from the literature.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            # Filter the graph
            filtered_graph, importance = _build_filtered_graph(
                raw_graph,
                max_nodes=params["max_nodes"],
                max_edges=params["max_edges"],
                min_importance=params["min_importance"],
            )

            # Extract key information
            nodes = list(filtered_graph.nodes())
            edges = list(filtered_graph.edges(data=True))

            # Find specific entity if requested
            entity1 = params["entity1"]
            entity2 = params["entity2"]

            entity1_info = None
            entity2_info = None
            entity1_neighbors = []
            entity2_neighbors = []
            shared_neighbors = []

            if entity1 and entity1 in filtered_graph:
                entity1_info = {
                    "label": entity1,
                    "importance": importance.get(entity1, 0.0),
                    "type": self._get_node_type(filtered_graph, entity1),
                    "degree": filtered_graph.degree(entity1),
                }
                entity1_neighbors = list(filtered_graph.neighbors(entity1))

            if entity2 and entity2 in filtered_graph:
                entity2_info = {
                    "label": entity2,
                    "importance": importance.get(entity2, 0.0),
                    "type": self._get_node_type(filtered_graph, entity2),
                    "degree": filtered_graph.degree(entity2),
                }
                entity2_neighbors = list(filtered_graph.neighbors(entity2))

            # Find shared neighbors if both entities exist
            if entity1 and entity2:
                set1 = set(entity1_neighbors)
                set2 = set(entity2_neighbors)
                shared_neighbors = list(set1 & set2)

            # Find direct path between entities
            path = None
            if entity1 and entity2 and entity1 in filtered_graph and entity2 in filtered_graph:
                try:
                    import networkx as nx
                    path = nx.shortest_path(filtered_graph, entity1, entity2)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path = None

            # Get embedding manager for semantic features
            manager = None
            if _get_embedding_manager:
                try:
                    manager = _get_embedding_manager()
                except Exception:
                    pass

            output = {
                "action": "query",
                "entity1": entity1,
                "entity2": entity2,
                "entity1_info": entity1_info,
                "entity2_info": entity2_info,
                "shared_neighbors": shared_neighbors[:10],
                "path": path,
                "total_nodes": filtered_graph.number_of_nodes(),
                "total_edges": filtered_graph.number_of_edges(),
                "top_entities": sorted(
                    [{"label": n, "importance": importance.get(n, 0.0)}
                     for n in nodes[:20]],
                    key=lambda x: x["importance"],
                    reverse=True,
                ),
                "entity1_neighbors": entity1_neighbors[:10],
                "entity2_neighbors": entity2_neighbors[:10],
                "has_embeddings": manager is not None,
            }

            confidence = ConfidenceScore(
                value=0.85,
                basis=f"Graph built from {len(results)} papers with {filtered_graph.number_of_nodes()} nodes and {filtered_graph.number_of_edges()} edges",
                source_module="knowledge_graphs",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Graph query failed: %s", exc)
            raise AgentExecutionError(
                f"Graph query failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_semantic_search(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a semantic search over the knowledge graph.

        Args:
            params: Query parameters.
            task: The task being executed.

        Returns:
            An AgentResult with semantic search results.

        Raises:
            AgentExecutionError: If the search fails.
        """
        query = params["semantic_query"]

        if not query:
            raise AgentExecutionError(
                "Semantic query is required for semantic search.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            # First build the graph
            import streamlit as st
            results = st.session_state.get("bx_last_results", [])

            if not results:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No literature search results available.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            pipeline = KnowledgeGraphPipeline()
            process_result = pipeline.process_papers(results)
            raw_graph = process_result.get("graph")

            if raw_graph is None or raw_graph.number_of_nodes() == 0:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No graph could be built from the literature.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            # Filter the graph
            filtered_graph, _ = _build_filtered_graph(
                raw_graph,
                max_nodes=params["max_nodes"],
                max_edges=params["max_edges"],
                min_importance=params["min_importance"],
            )

            # Get embedding manager
            if _get_embedding_manager is None:
                raise AgentExecutionError(
                    "Embedding manager not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            manager = _get_embedding_manager()
            if manager is None:
                raise AgentExecutionError(
                    "Failed to initialize embedding manager.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            # Sync entity embeddings
            if _sync_entity_embeddings:
                node_to_embedding_id = _sync_entity_embeddings(filtered_graph, manager)

                # Search using embeddings
                try:
                    results = manager.search(query, top_k=params["top_k"])
                except Exception as e:
                    self.logger.warning("Semantic search failed: %s", e)
                    results = []

                output = {
                    "action": "semantic",
                    "query": query,
                    "results": [
                        {
                            "entity": r.record.metadata.title,
                            "type": r.record.metadata.extra.get("entity_type", "unknown"),
                            "score": r.score,
                        }
                        for r in results
                    ],
                    "total_results": len(results),
                }

                confidence = ConfidenceScore(
                    value=0.8,
                    basis=f"Semantic search for '{query}' returned {len(results)} results",
                    source_module="knowledge_graphs",
                )

                return self._build_success_result(
                    task=task,
                    output=output,
                    confidence=confidence,
                )

            else:
                raise AgentExecutionError(
                    "Entity embedding sync not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Semantic search failed: %s", exc)
            raise AgentExecutionError(
                f"Semantic search failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_cluster(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute entity clustering over the knowledge graph.

        Args:
            params: Query parameters.
            task: The task being executed.

        Returns:
            An AgentResult with clustering results.

        Raises:
            AgentExecutionError: If clustering fails.
        """
        try:
            import streamlit as st
            results = st.session_state.get("bx_last_results", [])

            if not results:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No literature search results available.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            pipeline = KnowledgeGraphPipeline()
            process_result = pipeline.process_papers(results)
            raw_graph = process_result.get("graph")

            if raw_graph is None or raw_graph.number_of_nodes() == 0:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No graph could be built from the literature.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            filtered_graph, _ = _build_filtered_graph(
                raw_graph,
                max_nodes=params["max_nodes"],
                max_edges=params["max_edges"],
                min_importance=params["min_importance"],
            )

            if _get_embedding_manager is None or _sync_entity_embeddings is None:
                raise AgentExecutionError(
                    "Embedding features not available.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            manager = _get_embedding_manager()
            if manager is None:
                raise AgentExecutionError(
                    "Failed to initialize embedding manager.",
                    agent_type=self.agent_type,
                    is_recoverable=False,
                )

            node_to_embedding_id = _sync_entity_embeddings(filtered_graph, manager)

            threshold = params.get("threshold", 0.75)
            clusters = _cluster_entities_by_embedding(manager, node_to_embedding_id, threshold)

            output = {
                "action": "cluster",
                "threshold": threshold,
                "clusters": [
                    {"cluster_id": i, "entities": cluster, "size": len(cluster)}
                    for i, cluster in enumerate(clusters)
                    if len(cluster) > 1
                ],
                "total_clusters": len([c for c in clusters if len(c) > 1]),
                "total_entities": len(node_to_embedding_id),
            }

            confidence = ConfidenceScore(
                value=0.75,
                basis=f"Found {len(output['clusters'])} clusters at threshold {threshold}",
                source_module="knowledge_graphs",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Clustering failed: %s", exc)
            raise AgentExecutionError(
                f"Clustering failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    async def _execute_neighbors(self, params: dict[str, Any], task: AgentTask) -> AgentResult:
        """
        Execute a neighbor query to find entities connected to a given entity.

        Args:
            params: Query parameters.
            task: The task being executed.

        Returns:
            An AgentResult with neighbor information.

        Raises:
            AgentExecutionError: If the query fails.
        """
        entity = params["entity1"]

        if not entity:
            raise AgentExecutionError(
                "Entity name is required for neighbor query.",
                agent_type=self.agent_type,
                is_recoverable=False,
            )

        try:
            import streamlit as st
            results = st.session_state.get("bx_last_results", [])

            if not results:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No literature search results available.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            pipeline = KnowledgeGraphPipeline()
            process_result = pipeline.process_papers(results)
            raw_graph = process_result.get("graph")

            if raw_graph is None or raw_graph.number_of_nodes() == 0:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        "No graph could be built from the literature.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            filtered_graph, importance = _build_filtered_graph(
                raw_graph,
                max_nodes=params["max_nodes"],
                max_edges=params["max_edges"],
                min_importance=params["min_importance"],
            )

            if entity not in filtered_graph:
                return self._build_error_result(
                    task=task,
                    error=AgentExecutionError(
                        f"Entity '{entity}' not found in the knowledge graph.",
                        agent_type=self.agent_type,
                        is_recoverable=True,
                    ),
                    status=TaskStatus.FAILED,
                )

            neighbors = list(filtered_graph.neighbors(entity))

            # Get detailed neighbor info
            neighbor_info = []
            for n in neighbors[:20]:
                neighbor_info.append({
                    "entity": n,
                    "importance": importance.get(n, 0.0),
                    "type": self._get_node_type(filtered_graph, n),
                    "relation": self._get_edge_label(filtered_graph, entity, n),
                })

            output = {
                "action": "neighbors",
                "entity": entity,
                "degree": filtered_graph.degree(entity),
                "neighbor_count": len(neighbors),
                "neighbors": neighbor_info,
                "entity_importance": importance.get(entity, 0.0),
                "entity_type": self._get_node_type(filtered_graph, entity),
            }

            confidence = ConfidenceScore(
                value=0.9,
                basis=f"Found {len(neighbors)} neighbors for '{entity}'",
                source_module="knowledge_graphs",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except AgentError:
            raise
        except Exception as exc:
            self.logger.exception("Neighbor query failed for '%s': %s", entity, exc)
            raise AgentExecutionError(
                f"Neighbor query failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    def _get_node_type(self, graph, node: str) -> str:
        """
        Get the type of a node in the graph.

        Args:
            graph: The NetworkX graph.
            node: The node label.

        Returns:
            The node type as a string.
        """
        if hasattr(graph, 'nodes') and node in graph.nodes:
            data = graph.nodes[node]
            # Try common type keys
            for key in ("type", "entity_type", "label_type", "category", "ner_type"):
                if key in data:
                    val = data[key]
                    if val:
                        return str(val).strip().lower()
        return "unknown"

    def _get_edge_label(self, graph, source: str, target: str) -> str:
        """
        Get the label of an edge between two nodes.

        Args:
            graph: The NetworkX graph.
            source: The source node.
            target: The target node.

        Returns:
            The edge label as a string.
        """
        if graph.has_edge(source, target):
            data = graph.get_edge_data(source, target)
            if data:
                for key in ("relation", "relation_type", "label", "predicate", "type"):
                    if key in data:
                        val = data[key]
                        if val:
                            return str(val)
        return "connected_to"

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        This releases any resources acquired during initialization.
        """
        self.logger.info("Shutting down KnowledgeGraphAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "graph_query",
            "graph_semantic_search",
            "graph_cluster",
            "graph_neighbors",
            "graph_entity_lookup",
            "graph_relationship_analysis",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"KnowledgeGraphAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_knowledge_graph_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    default_max_nodes: int = 30,
    default_max_edges: int = 50,
    min_importance: float = 0.0,
) -> KnowledgeGraphAgent:
    """
    Factory function to create a KnowledgeGraphAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        default_max_nodes: Default maximum number of nodes to display.
        default_max_edges: Default maximum number of edges to display.
        min_importance: Minimum importance threshold for nodes.

    Returns:
        A configured KnowledgeGraphAgent instance.
    """
    return KnowledgeGraphAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        default_max_nodes=default_max_nodes,
        default_max_edges=default_max_edges,
        min_importance=min_importance,
    )


__all__: list[str] = [
    "KnowledgeGraphAgent",
    "create_knowledge_graph_agent",
]