"""
agents/planner_agent.py

Planner Agent for the BioNexus Multi-Agent AI System.

This agent is responsible for analyzing user queries, detecting intent,
and producing execution plans that determine which specialist agents
should be invoked and in what order. The planner supports both sequential
and parallel execution, handles dependencies between agents, and manages
ambiguous queries by requesting clarification.

The planner is designed to be:
    - Extensible: New agents and intents can be added easily
    - Rule-based: Initial implementation uses deterministic rules
    - LLM-ready: Designed so an LLM-based planner can replace the rule engine later
    - Context-aware: Preserves conversation context using Memory

Supported specialist agents:
    - LiteratureAgent: Biomedical literature search and analysis
    - DrugAgent: Drug information retrieval (PubChem)
    - SmilesAgent: Chemical structure analysis from SMILES
    - KnowledgeGraphAgent: Biomedical knowledge graph exploration
    - DrugDiscoveryAgent: AI-assisted drug discovery and candidate evaluation

Example:
    >>> agent = PlannerAgent()
    >>> await agent.start()
    >>> request = AgentRequest(
    ...     instruction="Find EGFR papers and predict drug candidates",
    ...     parameters={}
    ... )
    >>> response = await agent.handle_request(request)
    >>> # Response contains an execution plan with ordered agents

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
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

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------


class IntentCategory(str, Enum):
    """
    Categories of user intent that the planner can detect.

    Each intent corresponds to one or more specialist agents that can
    fulfill the user's request.
    """

    LITERATURE_SEARCH = "literature_search"
    DRUG_INFO = "drug_info"
    SMILES_ANALYSIS = "smiles_analysis"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DRUG_DISCOVERY = "drug_discovery"
    GENERAL_QA = "general_qa"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class ExecutionMode(str, Enum):
    """
    Execution mode for a plan step.

    SEQUENTIAL: Steps are executed one after another
    PARALLEL: Steps are executed concurrently
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


# ----------------------------------------------------------------------
# Planning models
# ----------------------------------------------------------------------


@dataclass
class PlanStep:
    """
    A single step in an execution plan.

    Attributes:
        step_id: Unique identifier for this step.
        agent_type: The type of agent to invoke.
        priority: Priority of this step (lower number = higher priority).
        dependencies: List of step IDs that must complete before this step.
        description: Human-readable description of what this step does.
        parameters: Parameters to pass to the agent.
        estimated_confidence: Confidence that this step is appropriate.
        is_parallel: Whether this step can run in parallel with others.
    """

    step_id: str
    agent_type: AgentType
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    estimated_confidence: float = 0.8
    is_parallel: bool = False

    def __post_init__(self) -> None:
        """Validate the step's configuration."""
        if not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string.")
        if self.priority < 0:
            raise ValueError("priority cannot be negative.")
        if not 0.0 <= self.estimated_confidence <= 1.0:
            raise ValueError("estimated_confidence must be within [0.0, 1.0].")


@dataclass
class ExecutionPlan:
    """
    An execution plan produced by the planner.

    Attributes:
        plan_id: Unique identifier for this plan.
        original_query: The user's original query.
        detected_intents: List of intents detected in the query.
        steps: The ordered list of plan steps.
        execution_mode: Whether steps should be sequential or parallel.
        requires_clarification: Whether the query is ambiguous and needs clarification.
        clarification_questions: List of questions to ask the user if clarification is needed.
        confidence: Overall confidence in the plan.
        explanation: Human-readable explanation of the plan.
    """

    plan_id: str
    original_query: str
    detected_intents: list[IntentCategory]
    steps: list[PlanStep]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    requires_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)
    confidence: float = 0.8
    explanation: str = ""

    def __post_init__(self) -> None:
        """Validate the plan's configuration."""
        if not self.plan_id.strip():
            raise ValueError("plan_id must be a non-empty string.")
        if not self.steps:
            raise ValueError("plan must contain at least one step.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0.0, 1.0].")

    def get_agent_types(self) -> list[AgentType]:
        """Return the list of agent types in the plan."""
        return [step.agent_type for step in self.steps]

    def get_sequential_steps(self) -> list[PlanStep]:
        """Return steps that must be executed sequentially."""
        return [s for s in self.steps if not s.is_parallel]

    def get_parallel_steps(self) -> list[PlanStep]:
        """Return steps that can be executed in parallel."""
        return [s for s in self.steps if s.is_parallel]


# ----------------------------------------------------------------------
# Agent capability registry
# ----------------------------------------------------------------------


@dataclass
class AgentCapability:
    """
    Capability definition for a specialist agent.

    Attributes:
        agent_type: The type of agent.
        intents: List of intents this agent can handle.
        keywords: Keywords that trigger this agent.
        patterns: Regex patterns that trigger this agent.
        description: Human-readable description of the agent's capabilities.
        requires_smiles: Whether this agent requires a SMILES string.
        requires_compound_name: Whether this agent requires a compound name.
        requires_disease: Whether this agent requires a disease name.
        priority: Default priority for this agent.
    """

    agent_type: AgentType
    intents: list[IntentCategory] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    description: str = ""
    requires_smiles: bool = False
    requires_compound_name: bool = False
    requires_disease: bool = False
    priority: int = 0


# ----------------------------------------------------------------------
# Planner configuration
# ----------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """
    Configuration for the Planner Agent.

    Attributes:
        default_confidence: Default confidence for routing decisions.
        min_confidence_threshold: Minimum confidence to execute a plan.
        ambiguous_threshold: Confidence below which queries are considered ambiguous.
        max_parallel_steps: Maximum number of steps to execute in parallel.
        enable_parallel_execution: Whether to enable parallel execution.
        enable_clarification: Whether to request clarification for ambiguous queries.
        default_timeout_seconds: Default timeout for plan execution.
    """

    default_confidence: float = 0.8
    min_confidence_threshold: float = 0.4
    ambiguous_threshold: float = 0.5
    max_parallel_steps: int = 3
    enable_parallel_execution: bool = True
    enable_clarification: bool = True
    default_timeout_seconds: float = 300.0


# ----------------------------------------------------------------------
# PlannerAgent
# ----------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    """
    Agent responsible for planning the execution of specialist agents.

    The PlannerAgent analyzes user queries, detects intents, and produces
    execution plans that determine which specialist agents to invoke and
    in what order. It supports both sequential and parallel execution,
    detects dependencies between agents, and handles ambiguous queries.

    Example:
        >>> agent = PlannerAgent()
        >>> await agent.start()
        >>> request = AgentRequest(
        ...     instruction="Find EGFR papers and predict drug candidates",
        ...     parameters={}
        ... )
        >>> response = await agent.handle_request(request)
        >>> plan = response.results[0].output["plan"]
        >>> print(plan.explanation)

    Attributes:
        memory: MemoryManager instance for conversation history.
        config: Planner configuration.
        _agent_capabilities: Registry of agent capabilities.
    """

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryManager] = None,
        planner_config: Optional[PlannerConfig] = None,
    ) -> None:
        """
        Initialize the Planner Agent.

        Args:
            agent_id: Unique identifier for this agent instance.
            config: Runtime configuration for this agent instance.
            memory: MemoryManager instance for conversation history.
            planner_config: Planner-specific configuration.
        """
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            agent_id=agent_id,
            config=config or AgentConfig(
                default_timeout_seconds=30.0,
                max_retries=2,
                enable_tool_use=False,
                enable_rag=False,
            ),
        )
        self.memory: MemoryManager = memory or MemoryManager()
        self.planner_config: PlannerConfig = planner_config or PlannerConfig()
        self._agent_capabilities: list[AgentCapability] = self._build_capability_registry()
        self._initialized = False

    def _build_capability_registry(self) -> list[AgentCapability]:
        """
        Build the registry of agent capabilities.

        This defines which intents each specialist agent can handle,
        what keywords trigger them, and their default priorities.

        Returns:
            A list of AgentCapability definitions.
        """
        return [
            AgentCapability(
                agent_type=AgentType.LITERATURE_SEARCH_AGENT,
                intents=[IntentCategory.LITERATURE_SEARCH],
                keywords=[
                    "paper", "papers", "publication", "publications",
                    "literature", "research", "study", "studies",
                    "journal", "articles", "article", "bibliography",
                    "find", "search", "bibliometric", "research article",
                    "meta-analysis", "clinical trial", "systematic review",
                    "finding papers", "literature search", "publication search",
                    "article search", "research articles", "scholarly articles",
                ],
                patterns=[
                    r"find\s+papers?",
                    r"search\s+for\s+papers?",
                    r"literature\s+search",
                    r"publication\s+search",
                    r"research\s+on\s+",
                    r"studies?\s+on\s+",
                    r"articles?\s+about\s+",
                    r"clinical\s+trials?\s+on\s+",
                    r"meta[- ]analysis",
                    r"systematic\s+review",
                ],
                description="Searches biomedical literature across multiple databases",
                priority=0,
            ),
            AgentCapability(
                agent_type=AgentType.DRUG_INFO_AGENT,
                intents=[IntentCategory.DRUG_INFO],
                keywords=[
                    "drug", "drugs", "medicine", "medicines",
                    "medication", "pharmaceutical", "pharmacology",
                    "compound", "compounds", "chemical", "chemicals",
                    "dosage", "dose", "indication", "contraindication",
                    "interaction", "side effect", "adverse effect",
                    "pubchem", "drug name", "drug information",
                    "pharmacodynamics", "pharmacokinetics", "pharmacokinetic",
                    "mechanism of action", "indications", "contraindications",
                ],
                patterns=[
                    r"drug\s+information",
                    r"about\s+drug",
                    r"tell\s+me\s+about\s+",
                    r"what\s+is\s+",
                    r"information\s+about\s+",
                    r"drug\s+name\s+",
                    r"molecular\s+properties",
                    r"PubChem",
                ],
                description="Retrieves comprehensive drug information from PubChem",
                requires_compound_name=True,
                priority=10,
            ),
            AgentCapability(
                agent_type=AgentType.SMILES_ANALYSIS_AGENT,
                intents=[IntentCategory.SMILES_ANALYSIS],
                keywords=[
                    "smiles", "structure", "molecular structure",
                    "chemical structure", "3d structure", "2d structure",
                    "molecule", "molecules", "chemical formula",
                    "functional group", "functional groups", "lipinski",
                    "rule of five", "atom", "atoms", "bond", "bonds",
                    "molecular weight", "logp", "tpsa", "h-bond",
                    "analyze structure", "chemical analysis",
                ],
                patterns=[
                    r"smiles\s+string",
                    r"molecular\s+structure",
                    r"chemical\s+structure",
                    r"3d\s+structure",
                    r"functional\s+groups?",
                    r"lipinski",
                    r"rule\s+of\s+five",
                    r"atom\s+information",
                    r"[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}",
                ],
                description="Analyzes chemical structures from SMILES strings",
                requires_smiles=True,
                priority=20,
            ),
            AgentCapability(
                agent_type=AgentType.KNOWLEDGE_GRAPH_AGENT,
                intents=[IntentCategory.KNOWLEDGE_GRAPH],
                keywords=[
                    "relationship", "relationships", "connection",
                    "connections", "network", "networks", "graph",
                    "node", "nodes", "edge", "edges", "pathway",
                    "gene", "genes", "protein", "proteins",
                    "disease", "diseases", "interaction", "interactions",
                    "knowledge graph", "biological network",
                    "gene-disease", "protein-protein", "drug-target",
                    "pathway analysis", "connectivity",
                ],
                patterns=[
                    r"relationship\s+between",
                    r"connection\s+between",
                    r"how\s+is\s+.*\s+related\s+to",
                    r"interacts?\s+with",
                    r"graph\s+of",
                    r"knowledge\s+graph",
                    r"pathways?\s+involving",
                    r"biological\s+network",
                    r"gene[- ]disease",
                    r"protein[- ]protein",
                ],
                description="Explores biomedical relationships and knowledge graphs",
                priority=30,
            ),
            AgentCapability(
                agent_type=AgentType.DRUG_REPURPOSING_AGENT,
                intents=[IntentCategory.DRUG_DISCOVERY],
                keywords=[
                    "drug discovery", "candidate", "candidates",
                    "similarity search", "similar compounds",
                    "target prediction", "toxicity prediction",
                    "repurposing", "drug repurposing", "repositioning",
                    "lead", "leads", "hit", "hits", "screening",
                    "virtual screening", "high-throughput screening",
                    "pharmacophore", "docking", "binding affinity",
                    "efficacy", "safety", "admet", "drug-likeness",
                    "rank", "ranking", "candidate ranking",
                ],
                patterns=[
                    r"drug\s+discovery",
                    r"find\s+drug",
                    r"similar\s+compounds?",
                    r"target\s+prediction",
                    r"toxicity\s+prediction",
                    r"repurposing",
                    r"candidate\s+ranking",
                    r"lead\s+compound",
                    r"virtual\s+screening",
                    r"drug[- ]likeness",
                    r"admet",
                ],
                description="Performs AI-assisted drug discovery and candidate evaluation",
                requires_smiles=True,
                priority=40,
            ),
        ]

    async def initialize(self) -> None:
        """
        Initialize the Planner Agent.

        This validates the agent capability registry.

        Raises:
            AgentInitializationError: If initialization fails.
        """
        self.logger.info(
            "PlannerAgent '%s' initialized with %d agent capabilities",
            self.agent_id,
            len(self._agent_capabilities),
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

    def _extract_planner_params(self, request: AgentRequest) -> dict[str, Any]:
        """
        Extract planner parameters from the request.

        Args:
            request: The inbound request.

        Returns:
            A dictionary of planner parameters.
        """
        params = request.parameters or {}

        return {
            "query": request.instruction,
            "conversation_id": request.conversation_id,
            "requested_by": request.requested_by,
            "metadata": request.metadata,
            "previous_results": params.get("previous_results", []),
            "preferred_agents": params.get("preferred_agents", []),
            "exclude_agents": params.get("exclude_agents", []),
            "force_agents": params.get("force_agents", []),
            "max_parallel": params.get("max_parallel", self.planner_config.max_parallel_steps),
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the planning task.

        Args:
            task: The task to execute.

        Returns:
            An AgentResult containing the execution plan.

        Raises:
            AgentExecutionError: If execution fails.
        """
        params = self._extract_planner_params(
            AgentRequest(
                request_id=task.request_id,
                instruction=task.description,
                parameters=task.input_data,
            )
        )

        try:
            plan = self._create_plan(
                query=params["query"],
                conversation_id=params["conversation_id"],
                previous_results=params["previous_results"],
                preferred_agents=params["preferred_agents"],
                exclude_agents=params["exclude_agents"],
                force_agents=params["force_agents"],
            )

            output = {
                "plan": plan,
                "plan_id": plan.plan_id,
                "detected_intents": [i.value for i in plan.detected_intents],
                "agent_count": len(plan.steps),
                "execution_mode": plan.execution_mode.value,
                "requires_clarification": plan.requires_clarification,
                "clarification_questions": plan.clarification_questions,
                "explanation": plan.explanation,
            }

            confidence = ConfidenceScore(
                value=plan.confidence,
                basis=f"Plan created with {len(plan.steps)} steps for {len(plan.detected_intents)} intents",
                source_module="planner",
            )

            return self._build_success_result(
                task=task,
                output=output,
                confidence=confidence,
            )

        except Exception as exc:
            self.logger.exception("Planning failed for query '%s': %s", params["query"], exc)
            raise AgentExecutionError(
                f"Planning failed: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

    def _create_plan(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        previous_results: list[Any] = None,
        preferred_agents: list[str] = None,
        exclude_agents: list[str] = None,
        force_agents: list[str] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for the given query.

        Args:
            query: The user's query.
            conversation_id: Optional conversation ID for context.
            previous_results: Previous results from the conversation.
            preferred_agents: Agents preferred by the user.
            exclude_agents: Agents to exclude from the plan.
            force_agents: Agents to force into the plan.

        Returns:
            An ExecutionPlan instance.
        """
        preferred_agents = preferred_agents or []
        exclude_agents = exclude_agents or []
        force_agents = force_agents or []

        # 1. Detect intents from the query
        detected_intents = self._detect_intents(query)

        if not detected_intents:
            # Try to detect from conversation context
            if previous_results:
                detected_intents = self._detect_intents_from_context(previous_results)

        if not detected_intents:
            detected_intents = [IntentCategory.UNKNOWN]

        # 2. Map intents to agents
        agent_matches = self._map_intents_to_agents(
            detected_intents,
            preferred_agents,
            exclude_agents,
            force_agents,
        )

        # 3. Create plan steps
        steps = self._create_steps(agent_matches, query)

        # 4. Detect dependencies
        steps = self._detect_dependencies(steps, query)

        # 5. Determine execution mode
        execution_mode = self._determine_execution_mode(steps)

        # 6. Check if clarification is needed
        requires_clarification = False
        clarification_questions = []

        if self.planner_config.enable_clarification:
            requires_clarification, clarification_questions = self._check_ambiguity(
                query, detected_intents, steps
            )

        # 7. Generate explanation
        explanation = self._generate_explanation(detected_intents, steps)

        # 8. Calculate confidence
        confidence = self._calculate_confidence(detected_intents, steps)

        return ExecutionPlan(
            plan_id=f"PLAN_{self.agent_id[:8]}_{len(detected_intents)}",
            original_query=query,
            detected_intents=detected_intents,
            steps=steps,
            execution_mode=execution_mode,
            requires_clarification=requires_clarification,
            clarification_questions=clarification_questions,
            confidence=confidence,
            explanation=explanation,
        )

    def _detect_intents(self, query: str) -> list[IntentCategory]:
        """
        Detect intents from the user's query.

        Uses keyword matching and regex patterns to identify which
        specialist agents should be invoked.

        Args:
            query: The user's query.

        Returns:
            A list of detected intents.
        """
        query_lower = query.lower()
        detected = set()

        for capability in self._agent_capabilities:
            # Check keywords
            for keyword in capability.keywords:
                if keyword.lower() in query_lower:
                    detected.update(capability.intents)
                    break

            # Check patterns
            for pattern in capability.patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected.update(capability.intents)
                    break

        # If no intents detected, try more specific detection
        if not detected:
            # Check for SMILES-like strings
            if re.search(r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}', query):
                detected.add(IntentCategory.SMILES_ANALYSIS)

            # Check for drug names (capitalized words)
            if re.search(r'\b[A-Z][a-z]{2,}\b', query):
                # Don't automatically add drug info - it could be a false positive
                # Only if the query is short or starts with "tell me about"
                if len(query.split()) <= 5 or re.search(r'^(tell|what|about)\s+(me\s+)?about', query_lower):
                    detected.add(IntentCategory.DRUG_INFO)

            # Check for literature indicators
            if any(word in query_lower for word in ["paper", "publication", "research", "study", "journal"]):
                detected.add(IntentCategory.LITERATURE_SEARCH)

        # Remove any unknown intents
        detected = {i for i in detected if i != IntentCategory.UNKNOWN}

        return list(detected) if detected else [IntentCategory.UNKNOWN]

    def _detect_intents_from_context(self, previous_results: list) -> list[IntentCategory]:
        """
        Detect intents from previous conversation results.

        Args:
            previous_results: Previous results from the conversation.

        Returns:
            A list of detected intents.
        """
        if not previous_results:
            return []

        detected = set()

        for result in previous_results:
            if hasattr(result, "output") and isinstance(result.output, dict):
                if "agent_type" in result.output:
                    agent_type = result.output["agent_type"]
                    if agent_type == "literature_search_agent":
                        detected.add(IntentCategory.LITERATURE_SEARCH)
                    elif agent_type == "drug_info_agent":
                        detected.add(IntentCategory.DRUG_INFO)
                    elif agent_type == "smiles_analysis_agent":
                        detected.add(IntentCategory.SMILES_ANALYSIS)
                    elif agent_type == "knowledge_graph_agent":
                        detected.add(IntentCategory.KNOWLEDGE_GRAPH)
                    elif agent_type in ["drug_repurposing_agent", "drug_discovery_agent"]:
                        detected.add(IntentCategory.DRUG_DISCOVERY)

        return list(detected)

    def _map_intents_to_agents(
        self,
        intents: list[IntentCategory],
        preferred: list[str],
        exclude: list[str],
        force: list[str],
    ) -> list[tuple[AgentType, IntentCategory, float]]:
        """
        Map intents to agent types with confidence scores.

        Args:
            intents: List of detected intents.
            preferred: Preferred agent types.
            exclude: Agent types to exclude.
            force: Agent types to force.

        Returns:
            A list of (agent_type, intent, confidence) tuples.
        """
        matches: list[tuple[AgentType, IntentCategory, float]] = []

        for intent in intents:
            if intent == IntentCategory.UNKNOWN:
                continue

            for capability in self._agent_capabilities:
                if intent in capability.intents:
                    agent_type_str = capability.agent_type.value

                    # Check if excluded
                    if agent_type_str in exclude:
                        continue

                    # Check if forced
                    if force and agent_type_str not in force:
                        continue

                    # Calculate confidence
                    confidence = self._calculate_routing_confidence(
                        capability, intent, preferred
                    )

                    matches.append((capability.agent_type, intent, confidence))

        # Remove duplicates, keeping the highest confidence match per agent
        seen_agents = set()
        unique_matches = []
        for agent_type, intent, confidence in sorted(matches, key=lambda x: -x[2]):
            if agent_type not in seen_agents:
                seen_agents.add(agent_type)
                unique_matches.append((agent_type, intent, confidence))

        # If force agents were specified, ensure they're included
        for agent_type_str in force:
            for capability in self._agent_capabilities:
                if capability.agent_type.value == agent_type_str:
                    # Find an intent that matches this agent
                    matching_intent = None
                    for intent in intents:
                        if intent in capability.intents:
                            matching_intent = intent
                            break
                    if matching_intent is None:
                        matching_intent = IntentCategory.GENERAL_QA

                    if capability.agent_type not in seen_agents:
                        unique_matches.append(
                            (capability.agent_type, matching_intent, 0.9)
                        )
                        seen_agents.add(capability.agent_type)

        return unique_matches

    def _calculate_routing_confidence(
        self,
        capability: AgentCapability,
        intent: IntentCategory,
        preferred: list[str],
    ) -> float:
        """
        Calculate the confidence for routing an intent to an agent.

        Args:
            capability: The agent capability.
            intent: The detected intent.
            preferred: Preferred agent types.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        confidence = self.planner_config.default_confidence

        # Boost confidence if the agent is preferred
        if capability.agent_type.value in preferred:
            confidence += 0.1

        # Boost confidence based on intent-agent match strength
        if intent in capability.intents:
            # If the intent is the first one for this agent, boost
            if capability.intents[0] == intent:
                confidence += 0.05

        # Cap confidence
        return min(1.0, confidence)

    def _create_steps(
        self,
        matches: list[tuple[AgentType, IntentCategory, float]],
        query: str,
    ) -> list[PlanStep]:
        """
        Create plan steps from agent matches.

        Args:
            matches: List of (agent_type, intent, confidence) tuples.
            query: The original query.

        Returns:
            A list of PlanStep instances.
        """
        steps = []

        for i, (agent_type, intent, confidence) in enumerate(matches):
            # Determine priority based on agent capability
            priority = 0
            for capability in self._agent_capabilities:
                if capability.agent_type == agent_type:
                    priority = capability.priority
                    break

            # Extract parameters based on agent type
            parameters = self._extract_agent_parameters(agent_type, query)

            step = PlanStep(
                step_id=f"STEP_{i+1}_{agent_type.value[:8]}",
                agent_type=agent_type,
                priority=priority,
                dependencies=[],
                description=f"Execute {agent_type.value.replace('_', ' ').title()} for intent: {intent.value}",
                parameters=parameters,
                estimated_confidence=confidence,
                is_parallel=self._can_run_parallel(agent_type, i, len(matches)),
            )
            steps.append(step)

        # Sort by priority
        steps.sort(key=lambda s: (s.priority, -s.estimated_confidence))

        return steps

    def _extract_agent_parameters(self, agent_type: AgentType, query: str) -> dict[str, Any]:
        """
        Extract parameters for a specific agent from the query.

        Args:
            agent_type: The agent type.
            query: The user's query.

        Returns:
            A dictionary of parameters.
        """
        params: dict[str, Any] = {}

        # Try to extract SMILES from query
        smiles_match = re.search(r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}', query)
        if smiles_match:
            params["smiles"] = smiles_match.group(0)

        # Try to extract compound name
        compound_match = re.search(r'(?:compound|drug|chemical)\s+([A-Z][a-z]{2,})', query, re.IGNORECASE)
        if compound_match:
            params["compound_name"] = compound_match.group(1)

        # Try to extract disease name
        disease_match = re.search(r'(?:disease|condition)\s+([A-Z][a-z\s]+)', query, re.IGNORECASE)
        if disease_match:
            params["disease_name"] = disease_match.group(1).strip()

        # Try to extract gene/protein name
        gene_match = re.search(r'([A-Z0-9]{2,8})\s+(?:gene|protein|mutation)', query, re.IGNORECASE)
        if gene_match:
            params["entity"] = gene_match.group(1)

        # Agent-specific parameter extraction
        if agent_type == AgentType.LITERATURE_SEARCH_AGENT:
            params["query"] = query
            params["max_results"] = 25

        elif agent_type == AgentType.DRUG_INFO_AGENT:
            if not params.get("compound_name"):
                # Try to get the main noun from the query
                words = query.split()
                for word in words:
                    if word[0].isupper() and len(word) > 2:
                        params["compound_name"] = word
                        break

        elif agent_type == AgentType.SMILES_ANALYSIS_AGENT:
            if not params.get("smiles"):
                params["smiles"] = query

        elif agent_type == AgentType.KNOWLEDGE_GRAPH_AGENT:
            params["entity1"] = params.get("entity", "")
            params["max_nodes"] = 30
            params["max_edges"] = 50

        elif agent_type in [AgentType.DRUG_REPURPOSING_AGENT, AgentType.DRUG_REPURPOSING_AGENT]:
            if not params.get("smiles"):
                params["smiles"] = query

        return params

    def _can_run_parallel(self, agent_type: AgentType, index: int, total: int) -> bool:
        """
        Determine if a step can run in parallel.

        Args:
            agent_type: The agent type.
            index: The step index.
            total: Total number of steps.

        Returns:
            True if the step can run in parallel.
        """
        # Can't run parallel if only one step
        if total <= 1:
            return False

        # Knowledge graph and literature search can run in parallel
        if agent_type in [
            AgentType.LITERATURE_SEARCH_AGENT,
            AgentType.KNOWLEDGE_GRAPH_AGENT,
        ]:
            return True

        # Independent agents can run in parallel
        if agent_type in [
            AgentType.DRUG_INFO_AGENT,
            AgentType.SMILES_ANALYSIS_AGENT,
        ]:
            return True

        # Drug discovery depends on prior analysis
        if agent_type in [AgentType.DRUG_REPURPOSING_AGENT, AgentType.DRUG_REPURPOSING_AGENT]:
            return False

        return False

    def _detect_dependencies(self, steps: list[PlanStep], query: str) -> list[PlanStep]:
        """
        Detect dependencies between steps.

        Args:
            steps: The list of plan steps.
            query: The user's query.

        Returns:
            The updated list of plan steps with dependencies.
        """
        step_map = {s.step_id: s for s in steps}
        agent_types = [s.agent_type for s in steps]

        for i, step in enumerate(steps):
            # Drug discovery depends on SMILES analysis
            if step.agent_type in [AgentType.DRUG_REPURPOSING_AGENT, AgentType.DRUG_REPURPOSING_AGENT]:
                # Find SMILES analysis step before this
                for j, other in enumerate(steps):
                    if j < i and other.agent_type == AgentType.SMILES_ANALYSIS_AGENT:
                        if other.step_id not in step.dependencies:
                            step.dependencies.append(other.step_id)

            # Knowledge graph depends on literature search
            if step.agent_type == AgentType.KNOWLEDGE_GRAPH_AGENT:
                for j, other in enumerate(steps):
                    if j < i and other.agent_type == AgentType.LITERATURE_SEARCH_AGENT:
                        if other.step_id not in step.dependencies:
                            step.dependencies.append(other.step_id)

        return steps

    def _determine_execution_mode(self, steps: list[PlanStep]) -> ExecutionMode:
        """
        Determine the execution mode based on steps.

        Args:
            steps: The list of plan steps.

        Returns:
            The execution mode.
        """
        if not self.planner_config.enable_parallel_execution:
            return ExecutionMode.SEQUENTIAL

        parallel_count = sum(1 for s in steps if s.is_parallel)
        if parallel_count > 1 and parallel_count <= self.planner_config.max_parallel_steps:
            return ExecutionMode.PARALLEL

        # If any step has dependencies, prefer sequential
        for step in steps:
            if step.dependencies:
                return ExecutionMode.SEQUENTIAL

        return ExecutionMode.SEQUENTIAL

    def _check_ambiguity(
        self,
        query: str,
        intents: list[IntentCategory],
        steps: list[PlanStep],
    ) -> tuple[bool, list[str]]:
        """
        Check if the query is ambiguous and needs clarification.

        Args:
            query: The user's query.
            intents: Detected intents.
            steps: The plan steps.

        Returns:
            A tuple of (requires_clarification, clarification_questions).
        """
        questions = []

        # Check if multiple intents detected
        if len(intents) > 1:
            questions.append(
                f"I notice your query involves multiple topics: {', '.join(i.value for i in intents)}. "
                "Which area would you like me to focus on first?"
            )

        # Check if no intents detected
        if len(intents) == 1 and intents[0] == IntentCategory.UNKNOWN:
            questions.append(
                "I'm not sure which area you're asking about. "
                "Are you looking for literature search, drug information, chemical analysis, "
                "knowledge graph exploration, or drug discovery?"
            )

        # Check if required parameters are missing for the planned agents
        for step in steps:
            if step.agent_type == AgentType.SMILES_ANALYSIS_AGENT:
                if not step.parameters.get("smiles"):
                    questions.append("I need a SMILES string to analyze the chemical structure. Could you provide one?")
            if step.agent_type == AgentType.DRUG_INFO_AGENT:
                if not step.parameters.get("compound_name"):
                    questions.append("Could you specify which drug or compound you'd like information about?")
            if step.agent_type in [AgentType.DRUG_REPURPOSING_AGENT, AgentType.DRUG_REPURPOSING_AGENT]:
                if not step.parameters.get("smiles"):
                    questions.append("I need a SMILES string for drug discovery analysis. Could you provide one?")

        # Limit to 3 questions
        questions = questions[:3]

        return len(questions) > 0, questions

    def _generate_explanation(
        self,
        intents: list[IntentCategory],
        steps: list[PlanStep],
    ) -> str:
        """
        Generate a human-readable explanation of the plan.

        Args:
            intents: Detected intents.
            steps: The plan steps.

        Returns:
            A human-readable explanation string.
        """
        if not steps:
            return "No plan could be generated for this query."

        intent_names = [i.value.replace("_", " ").title() for i in intents if i != IntentCategory.UNKNOWN]
        intent_descriptions = ", ".join(intent_names) if intent_names else "general query"

        lines = [
            f"Based on your query, I've detected the following intent(s): {intent_descriptions}.",
            "",
            "I will execute the following steps:",
        ]

        for i, step in enumerate(steps, 1):
            agent_name = step.agent_type.value.replace("_", " ").title()
            lines.append(f"  {i}. {agent_name} (confidence: {step.estimated_confidence:.0%})")
            if step.dependencies:
                deps = ", ".join(step.dependencies)
                lines.append(f"     Depends on: {deps}")
            if step.is_parallel:
                lines.append("     (Can run in parallel)")
            if step.parameters:
                param_str = ", ".join(f"{k}={v}" for k, v in step.parameters.items() if v)
                if param_str:
                    lines.append(f"     Parameters: {param_str}")

        execution_mode = "parallel" if steps and all(s.is_parallel for s in steps) else "sequential"
        lines.append("")
        lines.append(f"Execution mode: {execution_mode}")

        return "\n".join(lines)

    def _calculate_confidence(
        self,
        intents: list[IntentCategory],
        steps: list[PlanStep],
    ) -> float:
        """
        Calculate overall confidence in the plan.

        Args:
            intents: Detected intents.
            steps: The plan steps.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        if not steps:
            return 0.0

        # Start with average step confidence
        avg_step_confidence = sum(s.estimated_confidence for s in steps) / len(steps)

        # Adjust for intent coverage
        if intents and intents[0] == IntentCategory.UNKNOWN:
            avg_step_confidence *= 0.6

        # Adjust for number of steps
        if len(steps) > 3:
            avg_step_confidence *= 0.9

        return min(1.0, avg_step_confidence)

    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.
        """
        self.logger.info("Shutting down PlannerAgent '%s'...", self.agent_id)
        self._initialized = False

    def get_capabilities(self) -> list[str]:
        """
        Return the capabilities of this agent.

        Returns:
            A list of capability identifiers.
        """
        return [
            "planning",
            "intent_detection",
            "agent_routing",
            "dependency_analysis",
            "clarification",
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation for logging/debugging."""
        return (
            f"PlannerAgent(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_planner_agent(
    *,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryManager] = None,
    planner_config: Optional[PlannerConfig] = None,
) -> PlannerAgent:
    """
    Factory function to create a PlannerAgent instance.

    Args:
        agent_id: Unique identifier for this agent instance.
        config: Runtime configuration for this agent instance.
        memory: MemoryManager instance for conversation history.
        planner_config: Planner-specific configuration.

    Returns:
        A configured PlannerAgent instance.
    """
    return PlannerAgent(
        agent_id=agent_id,
        config=config,
        memory=memory,
        planner_config=planner_config,
    )


__all__: list[str] = [
    "IntentCategory",
    "ExecutionMode",
    "PlanStep",
    "ExecutionPlan",
    "AgentCapability",
    "PlannerConfig",
    "PlannerAgent",
    "create_planner_agent",
]