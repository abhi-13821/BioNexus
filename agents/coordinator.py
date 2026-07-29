"""
agents/coordinator.py

Coordinator Agent for the BioNexus Multi-Agent AI System.

The Coordinator is the central orchestration engine of the BioNexus
Multi-Agent AI system. It receives user queries, invokes the PlannerAgent
to obtain an execution plan, executes the required specialist agents
(sequentially or in parallel), merges outputs, and returns a unified
response.

The Coordinator is designed to be:
    - Extensible: New specialist agents can be added via dependency injection
    - Thread-safe: Designed for future asynchronous execution
    - Robust: Handles agent failures, timeouts, and partial execution gracefully
    - Observable: Collects comprehensive execution metadata
    - SOLID-compliant: Clean separation of concerns

Responsibilities:
    1. Receive user queries
    2. Invoke PlannerAgent for execution plans
    3. Execute specialist agents (sequential/parallel)
    4. Maintain shared Memory instance
    5. Collect execution metadata
    6. Merge outputs into unified responses
    7. Handle exceptions gracefully
    8. Return structured CoordinatorResponse

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Type

from agents.base_agent import (
    AgentConfig,
    AgentError,
    AgentExecutionError,
    BaseAgent,
)
from agents.drug_agent import DrugAgent
from agents.drug_discovery_agent import DrugDiscoveryAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.literature_agent import LiteratureAgent
from agents.memory import MemoryManager
from agents.models import (
    AgentRequest,
    AgentResponse,
    AgentResult,
    AgentTask,
    AgentType,
    ConfidenceScore,
    ErrorInformation,
    ExecutionMetadata,
    TaskStatus,
)
from agents.planner_agent import (
    ExecutionMode,
    ExecutionPlan,
    PlannerAgent,
    PlanStep,
)
from agents.smiles_agent import SmilesAgent

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Coordinator-specific exceptions
# ----------------------------------------------------------------------


class CoordinatorError(Exception):
    """Base exception for Coordinator errors."""

    pass


class CoordinatorInitializationError(CoordinatorError):
    """Raised when the Coordinator fails to initialize."""

    pass


class CoordinatorExecutionError(CoordinatorError):
    """Raised when execution of a plan fails."""

    pass


class AgentNotFoundError(CoordinatorError):
    """Raised when a requested agent is not registered."""

    pass


class AgentTimeoutError(CoordinatorError):
    """Raised when an agent execution times out."""

    pass


# ----------------------------------------------------------------------
# Coordinator configuration
# ----------------------------------------------------------------------


@dataclass
class CoordinatorConfig:
    """
    Configuration for the Coordinator.

    Attributes:
        default_timeout_seconds: Default timeout for agent execution.
        max_retries: Maximum number of retries for failed agents.
        retry_delay_seconds: Delay between retries.
        enable_parallel_execution: Whether to enable parallel execution.
        max_parallel_tasks: Maximum number of parallel tasks.
        enable_partial_results: Whether to return partial results on failure.
        collect_timing: Whether to collect timing metadata.
        agent_timeout_seconds: Per-agent timeout override.
    """

    default_timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    enable_parallel_execution: bool = True
    max_parallel_tasks: int = 5
    enable_partial_results: bool = True
    collect_timing: bool = True
    agent_timeout_seconds: dict[AgentType, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be positive.")


# ----------------------------------------------------------------------
# Coordinator response models
# ----------------------------------------------------------------------


@dataclass
class AgentExecutionResult:
    """
    Result of executing a single agent.

    Attributes:
        agent_type: The type of agent that was executed.
        agent_id: The specific agent instance ID.
        success: Whether execution was successful.
        result: The AgentResult from the execution.
        error: Error information if execution failed.
        start_time: When execution started.
        end_time: When execution completed.
        duration_ms: Execution duration in milliseconds.
        retries: Number of retries attempted.
    """

    agent_type: AgentType
    agent_id: str
    success: bool
    result: Optional[AgentResult] = None
    error: Optional[ErrorInformation] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retries: int = 0

    def __post_init__(self) -> None:
        """Calculate duration if end_time is set."""
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000.0


@dataclass
class CoordinatorResponse:
    """
    The unified response from the Coordinator.

    Attributes:
        response_id: Unique identifier for this response.
        request_id: Identifier of the original request.
        query: The original user query.
        status: Overall status of the execution.
        message: Human-readable summary message.
        plan: The execution plan that was used.
        results: Results from each agent execution.
        merged_output: Merged output from all agents.
        confidence: Overall confidence in the response.
        errors: List of errors encountered.
        warnings: List of warnings encountered.
        execution_metadata: Execution metadata for the coordinator.
        agent_results: Detailed results per agent.
        created_at: When the response was created.
    """

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    query: str = ""
    status: TaskStatus = TaskStatus.COMPLETED
    message: str = ""
    plan: Optional[ExecutionPlan] = None
    results: list[AgentResult] = field(default_factory=list)
    merged_output: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[ConfidenceScore] = None
    errors: list[ErrorInformation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_metadata: Optional[ExecutionMetadata] = None
    agent_results: list[AgentExecutionResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_success(self) -> bool:
        """Return whether the overall execution was successful."""
        return self.status == TaskStatus.COMPLETED

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Return the total execution duration in milliseconds."""
        if self.execution_metadata and self.execution_metadata.duration_ms:
            return self.execution_metadata.duration_ms
        return None


# ----------------------------------------------------------------------
# Agent registry
# ----------------------------------------------------------------------


class AgentRegistry:
    """
    Registry for specialist agents.

    This class manages the registration and retrieval of agent classes
    and instances. It supports dependency injection for adding new agents.
    """

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agent_classes: dict[AgentType, Type[BaseAgent]] = {}
        self._agent_instances: dict[AgentType, BaseAgent] = {}
        self._agent_factories: dict[AgentType, Callable[[], BaseAgent]] = {}
        self._initialized: bool = False

    def register(
        self,
        agent_type: AgentType,
        agent_class: Optional[Type[BaseAgent]] = None,
        factory: Optional[Callable[[], BaseAgent]] = None,
    ) -> None:
        """
        Register an agent type with its class or factory.

        Args:
            agent_type: The AgentType to register.
            agent_class: The agent class (optional if factory is provided).
            factory: A factory function that returns an agent instance.

        Raises:
            ValueError: If neither agent_class nor factory is provided.
        """
        if agent_class is None and factory is None:
            raise ValueError("Either agent_class or factory must be provided.")

        if agent_class:
            self._agent_classes[agent_type] = agent_class

        if factory:
            self._agent_factories[agent_type] = factory

        # Remove any existing instance so it gets recreated with new class/factory
        if agent_type in self._agent_instances:
            del self._agent_instances[agent_type]

        logger.info("Registered agent type: %s", agent_type.value)

    def get_instance(self, agent_type: AgentType, config: Optional[AgentConfig] = None) -> BaseAgent:
        """
        Get or create an agent instance of the specified type.

        Args:
            agent_type: The type of agent to get.
            config: Optional configuration for the agent.

        Returns:
            An instance of the requested agent.

        Raises:
            AgentNotFoundError: If the agent type is not registered.
        """
        # Check if we have a cached instance
        if agent_type in self._agent_instances:
            return self._agent_instances[agent_type]

        # Try to create from factory
        if agent_type in self._agent_factories:
            instance = self._agent_factories[agent_type]()
            self._agent_instances[agent_type] = instance
            return instance

        # Try to create from class
        if agent_type in self._agent_classes:
            agent_class = self._agent_classes[agent_type]
            instance = agent_class(config=config)
            self._agent_instances[agent_type] = instance
            return instance

        raise AgentNotFoundError(f"Agent type '{agent_type.value}' is not registered.")

    def has_agent(self, agent_type: AgentType) -> bool:
        """
        Check if an agent type is registered.

        Args:
            agent_type: The agent type to check.

        Returns:
            True if the agent type is registered.
        """
        return agent_type in self._agent_classes or agent_type in self._agent_factories

    def list_agents(self) -> list[AgentType]:
        """
        List all registered agent types.

        Returns:
            A list of registered AgentType values.
        """
        return list(set(self._agent_classes.keys()) | set(self._agent_factories.keys()))

    def clear_instances(self) -> None:
        """Clear all cached agent instances."""
        self._agent_instances.clear()

    async def initialize_all(self) -> None:
        """Initialize all registered agents."""
        for agent_type in self.list_agents():
            try:
                instance = self.get_instance(agent_type)
                if not instance.is_initialized:
                    await instance.start()
            except Exception as e:
                logger.warning("Failed to initialize agent %s: %s", agent_type.value, e)

    async def shutdown_all(self) -> None:
        """Shut down all registered agents."""
        for agent_type, instance in self._agent_instances.items():
            try:
                if instance.is_initialized:
                    await instance.stop()
            except Exception as e:
                logger.warning("Failed to shut down agent %s: %s", agent_type.value, e)


# ----------------------------------------------------------------------
# Default agent factory
# ----------------------------------------------------------------------


def default_agent_factory() -> AgentRegistry:
    """
    Create a default AgentRegistry with all specialist agents registered.

    Returns:
        A configured AgentRegistry instance.
    """
    registry = AgentRegistry()

    # Register all specialist agents
    registry.register(
        AgentType.LITERATURE_SEARCH_AGENT,
        agent_class=LiteratureAgent,
    )
    registry.register(
        AgentType.DRUG_INFO_AGENT,
        agent_class=DrugAgent,
    )
    registry.register(
        AgentType.SMILES_ANALYSIS_AGENT,
        agent_class=SmilesAgent,
    )
    registry.register(
        AgentType.KNOWLEDGE_GRAPH_AGENT,
        agent_class=KnowledgeGraphAgent,
    )
    registry.register(
        AgentType.DRUG_REPURPOSING_AGENT,
        agent_class=DrugDiscoveryAgent,
    )

    # Register the planner
    registry.register(
        AgentType.ORCHESTRATOR,
        agent_class=PlannerAgent,
    )

    return registry


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------


class Coordinator:
    """
    Central orchestration engine for the BioNexus Multi-Agent AI system.

    The Coordinator receives user queries, invokes the PlannerAgent to
    obtain execution plans, executes specialist agents, and returns
    unified responses.

    Example:
        >>> coordinator = Coordinator()
        >>> response = await coordinator.process_query(
        ...     query="Find EGFR papers and predict drug candidates",
        ...     conversation_id="conv_123"
        ... )
        >>> print(response.message)

    Attributes:
        config: Coordinator configuration.
        registry: Agent registry containing all specialist agents.
        memory: Shared MemoryManager instance.
        planner: PlannerAgent instance.
    """

    def __init__(
        self,
        config: Optional[CoordinatorConfig] = None,
        registry: Optional[AgentRegistry] = None,
        memory: Optional[MemoryManager] = None,
        planner: Optional[PlannerAgent] = None,
    ) -> None:
        """
        Initialize the Coordinator.

        Args:
            config: Coordinator configuration.
            registry: Agent registry. If not provided, a default registry is used.
            memory: Shared MemoryManager. If not provided, a new instance is created.
            planner: PlannerAgent instance. If not provided, one is created.

        Raises:
            CoordinatorInitializationError: If initialization fails.
        """
        self.config: CoordinatorConfig = config or CoordinatorConfig()
        self.registry: AgentRegistry = registry or default_agent_factory()
        self.memory: MemoryManager = memory or MemoryManager()
        self._initialized: bool = False

        # Initialize the planner
        if planner is None:
            planner_config = AgentConfig(
                default_timeout_seconds=30.0,
                max_retries=2,
            )
            self.planner = PlannerAgent(
                agent_id="planner_primary",
                config=planner_config,
                memory=self.memory,
            )
        else:
            self.planner = planner

        self._logger = logging.getLogger(f"{__name__}.Coordinator")

    async def initialize(self) -> None:
        """
        Initialize the Coordinator and all registered agents.

        Raises:
            CoordinatorInitializationError: If initialization fails.
        """
        if self._initialized:
            return

        try:
            # Initialize the planner
            await self.planner.start()

            # Initialize all registered agents
            await self.registry.initialize_all()

            self._initialized = True
            self._logger.info(
                "Coordinator initialized with %d agents",
                len(self.registry.list_agents()),
            )

        except Exception as e:
            self._logger.error("Coordinator initialization failed: %s", e)
            raise CoordinatorInitializationError(f"Initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shut down the Coordinator and all registered agents."""
        self._logger.info("Shutting down Coordinator...")

        # Shut down all agents
        await self.registry.shutdown_all()

        # Shut down the planner
        await self.planner.stop()

        self._initialized = False
        self._logger.info("Coordinator shut down successfully.")

    async def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        requested_by: Optional[str] = None,
    ) -> CoordinatorResponse:
        """
        Process a user query through the multi-agent system.

        This is the main entry point for the Coordinator.

        Args:
            query: The user's natural language query.
            conversation_id: Optional conversation ID for context.
            parameters: Optional additional parameters.
            requested_by: Optional identifier of the requester.

        Returns:
            A CoordinatorResponse containing the unified result.

        Raises:
            CoordinatorError: If processing fails catastrophically.
        """
        if not self._initialized:
            raise CoordinatorInitializationError(
                "Coordinator is not initialized. Call initialize() first."
            )

        request_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        parameters = parameters or {}

        self._logger.info(
            "Processing query: '%s' (request_id=%s, conversation_id=%s)",
            query[:100],
            request_id,
            conversation_id,
        )

        try:
            # Step 1: Get the execution plan from the planner
            plan = await self._get_plan(query, conversation_id, parameters, requested_by)

            if plan.requires_clarification and plan.clarification_questions:
                return self._create_clarification_response(
                    request_id, query, plan, conversation_id
                )

            # Step 2: Execute the plan
            execution_results = await self._execute_plan(plan, conversation_id, parameters)

            # Step 3: Merge outputs
            merged_output = self._merge_outputs(execution_results)

            # Step 4: Build the response
            response = self._build_response(
                request_id=request_id,
                query=query,
                plan=plan,
                execution_results=execution_results,
                merged_output=merged_output,
                start_time=start_time,
                conversation_id=conversation_id,
            )

            self._logger.info(
                "Query processed successfully: %s (status=%s, agents=%d)",
                request_id,
                response.status.value,
                len(response.agent_results),
            )

            return response

        except Exception as e:
            self._logger.error("Query processing failed for %s: %s", request_id, e)
            return self._create_error_response(
                request_id=request_id,
                query=query,
                error=e,
                start_time=start_time,
                conversation_id=conversation_id,
            )

    async def _get_plan(
        self,
        query: str,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
        requested_by: Optional[str],
    ) -> ExecutionPlan:
        """
        Get an execution plan from the PlannerAgent.

        Args:
            query: The user's query.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.
            requested_by: Optional requester identifier.

        Returns:
            An ExecutionPlan.

        Raises:
            CoordinatorExecutionError: If the planner fails.
        """
        try:
            request = AgentRequest(
                request_id=str(uuid.uuid4()),
                instruction=query,
                agent_type=AgentType.ORCHESTRATOR,
                parameters=parameters,
                conversation_id=conversation_id,
                requested_by=requested_by,
            )

            response = await self.planner.handle_request(request)

            if response.status == TaskStatus.FAILED:
                error_messages = [e.message for e in response.errors]
                raise CoordinatorExecutionError(
                    f"Planner failed: {'; '.join(error_messages)}"
                )

            # Extract the plan from the response
            if response.results and response.results[0].output:
                plan = response.results[0].output.get("plan")
                if plan and isinstance(plan, ExecutionPlan):
                    return plan

            raise CoordinatorExecutionError("Planner did not return a valid execution plan.")

        except CoordinatorExecutionError:
            raise
        except Exception as e:
            raise CoordinatorExecutionError(f"Failed to get execution plan: {e}") from e

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
    ) -> list[AgentExecutionResult]:
        """
        Execute an execution plan.

        Args:
            plan: The execution plan to execute.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.

        Returns:
            A list of AgentExecutionResult instances.

        Raises:
            CoordinatorExecutionError: If execution fails catastrophically.
        """
        results: list[AgentExecutionResult] = []

        if plan.execution_mode == ExecutionMode.PARALLEL:
            # Execute steps in parallel
            results = await self._execute_parallel(plan, conversation_id, parameters)
        else:
            # Execute steps sequentially
            results = await self._execute_sequential(plan, conversation_id, parameters)

        return results

    async def _execute_sequential(
        self,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
    ) -> list[AgentExecutionResult]:
        """
        Execute plan steps sequentially.

        Args:
            plan: The execution plan.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.

        Returns:
            A list of AgentExecutionResult instances.
        """
        results: list[AgentExecutionResult] = []
        completed_step_ids = set()

        for step in plan.steps:
            # Check if dependencies are met
            if not self._dependencies_met(step, completed_step_ids):
                # Skip step if dependencies not met
                continue

            result = await self._execute_step(step, plan, conversation_id, parameters)
            results.append(result)

            if result.success:
                completed_step_ids.add(step.step_id)

            # If a critical step fails and we don't want partial results
            if not result.success and not self.config.enable_partial_results:
                break

        return results

    async def _execute_parallel(
        self,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
    ) -> list[AgentExecutionResult]:
        """
        Execute plan steps in parallel.

        Args:
            plan: The execution plan.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.

        Returns:
            A list of AgentExecutionResult instances.
        """
        # Determine which steps can run in parallel
        parallel_steps = [
            s for s in plan.steps
            if s.is_parallel and not s.dependencies
        ]

        # Determine sequential steps (those with dependencies)
        sequential_steps = [
            s for s in plan.steps
            if not s.is_parallel or s.dependencies
        ]

        results: list[AgentExecutionResult] = []
        completed_step_ids = set()

        # Run parallel steps
        if parallel_steps and self.config.enable_parallel_execution:
            # Limit parallel tasks
            parallel_steps = parallel_steps[:self.config.max_parallel_tasks]

            # Run parallel steps
            parallel_tasks = [
                self._execute_step_with_timeout(step, plan, conversation_id, parameters)
                for step in parallel_steps
            ]

            # Wait for all parallel tasks to complete
            step_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

            for step, step_result in zip(parallel_steps, step_results):
                if isinstance(step_result, Exception):
                    # Handle exception
                    result = AgentExecutionResult(
                        agent_type=step.agent_type,
                        agent_id="unknown",
                        success=False,
                        error=ErrorInformation(
                            error_type="ExecutionError",
                            message=str(step_result),
                        ),
                    )
                else:
                    result = step_result

                results.append(result)
                if result.success:
                    completed_step_ids.add(step.step_id)

        # Run sequential steps
        for step in sequential_steps:
            # Check if dependencies are met
            if not self._dependencies_met(step, completed_step_ids):
                continue

            result = await self._execute_step(step, plan, conversation_id, parameters)
            results.append(result)

            if result.success:
                completed_step_ids.add(step.step_id)

            if not result.success and not self.config.enable_partial_results:
                break

        return results

    async def _execute_step(
        self,
        step: PlanStep,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
    ) -> AgentExecutionResult:
        """
        Execute a single plan step.

        Args:
            step: The plan step to execute.
            plan: The execution plan.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.

        Returns:
            An AgentExecutionResult.
        """
        start_time = datetime.now(timezone.utc)
        retries = 0
        last_error = None

        # Get the agent
        try:
            agent = self.registry.get_instance(step.agent_type)
        except AgentNotFoundError as e:
            return AgentExecutionResult(
                agent_type=step.agent_type,
                agent_id="unknown",
                success=False,
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                error=ErrorInformation(
                    error_type="AgentNotFoundError",
                    message=str(e),
                    agent_type=step.agent_type,
                ),
                retries=0,
            )

        # Prepare the request
        request = AgentRequest(
            request_id=plan.plan_id,
            instruction=step.description,
            agent_type=step.agent_type,
            parameters={**step.parameters, **parameters},
            conversation_id=conversation_id,
            priority=step.priority,
        )

        # Execute with retries
        while retries <= self.config.max_retries:
            try:
                start_time = datetime.now(timezone.utc)

                # Execute the agent
                response = await agent.handle_request(request)

                end_time = datetime.now(timezone.utc)

                # Extract the result
                if response.results:
                    result = response.results[0]
                else:
                    result = None

                if response.status == TaskStatus.COMPLETED:
                    return AgentExecutionResult(
                        agent_type=step.agent_type,
                        agent_id=agent.agent_id,
                        success=True,
                        result=result,
                        start_time=start_time,
                        end_time=end_time,
                        retries=retries,
                    )
                else:
                    # Failed but we can retry
                    last_error = response.errors[0] if response.errors else None
                    retries += 1
                    if retries <= self.config.max_retries:
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    else:
                        return AgentExecutionResult(
                            agent_type=step.agent_type,
                            agent_id=agent.agent_id,
                            success=False,
                            result=result,
                            start_time=start_time,
                            end_time=end_time,
                            error=last_error,
                            retries=retries,
                        )

            except asyncio.TimeoutError:
                last_error = ErrorInformation(
                    error_type="TimeoutError",
                    message=f"Agent execution timed out after {self.config.default_timeout_seconds}s",
                    agent_type=step.agent_type,
                )
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                else:
                    return AgentExecutionResult(
                        agent_type=step.agent_type,
                        agent_id="unknown",
                        success=False,
                        start_time=start_time,
                        end_time=datetime.now(timezone.utc),
                        error=last_error,
                        retries=retries,
                    )

            except Exception as e:
                last_error = ErrorInformation(
                    error_type=type(e).__name__,
                    message=str(e),
                    agent_type=step.agent_type,
                )
                retries += 1
                if retries <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                else:
                    return AgentExecutionResult(
                        agent_type=step.agent_type,
                        agent_id="unknown",
                        success=False,
                        start_time=start_time,
                        end_time=datetime.now(timezone.utc),
                        error=last_error,
                        retries=retries,
                    )

        # Should never reach here
        return AgentExecutionResult(
            agent_type=step.agent_type,
            agent_id="unknown",
            success=False,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            error=ErrorInformation(
                error_type="UnknownError",
                message="Execution failed for unknown reasons.",
                agent_type=step.agent_type,
            ),
            retries=retries,
        )

    async def _execute_step_with_timeout(
        self,
        step: PlanStep,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
        parameters: dict[str, Any],
    ) -> AgentExecutionResult:
        """
        Execute a step with a timeout.

        Args:
            step: The plan step.
            plan: The execution plan.
            conversation_id: Optional conversation ID.
            parameters: Additional parameters.

        Returns:
            An AgentExecutionResult.
        """
        timeout = self.config.agent_timeout_seconds.get(
            step.agent_type,
            self.config.default_timeout_seconds,
        )

        try:
            return await asyncio.wait_for(
                self._execute_step(step, plan, conversation_id, parameters),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return AgentExecutionResult(
                agent_type=step.agent_type,
                agent_id="unknown",
                success=False,
                error=ErrorInformation(
                    error_type="TimeoutError",
                    message=f"Agent execution timed out after {timeout}s",
                    agent_type=step.agent_type,
                ),
                retries=0,
            )

    def _dependencies_met(self, step: PlanStep, completed_ids: set[str]) -> bool:
        """
        Check if a step's dependencies are met.

        Args:
            step: The plan step.
            completed_ids: Set of completed step IDs.

        Returns:
            True if all dependencies are met.
        """
        return all(dep in completed_ids for dep in step.dependencies)

    def _merge_outputs(self, results: list[AgentExecutionResult]) -> dict[str, Any]:
        """
        Merge outputs from multiple agent executions.

        Args:
            results: List of agent execution results.

        Returns:
            A dictionary containing merged outputs.
        """
        merged: dict[str, Any] = {
            "agent_outputs": {},
            "total_agents": len(results),
            "successful_agents": sum(1 for r in results if r.success),
            "failed_agents": sum(1 for r in results if not r.success),
        }

        for result in results:
            agent_key = result.agent_type.value
            if result.success and result.result:
                merged["agent_outputs"][agent_key] = {
                    "success": True,
                    "output": result.result.output,
                    "confidence": result.result.confidence,
                }
            else:
                merged["agent_outputs"][agent_key] = {
                    "success": False,
                    "error": result.error.message if result.error else "Unknown error",
                }

        return merged

    def _build_response(
        self,
        request_id: str,
        query: str,
        plan: ExecutionPlan,
        execution_results: list[AgentExecutionResult],
        merged_output: dict[str, Any],
        start_time: datetime,
        conversation_id: Optional[str],
    ) -> CoordinatorResponse:
        """
        Build a CoordinatorResponse from execution results.

        Args:
            request_id: The request ID.
            query: The original query.
            plan: The execution plan.
            execution_results: Results from agent executions.
            merged_output: Merged output from all agents.
            start_time: When execution started.
            conversation_id: Optional conversation ID.

        Returns:
            A CoordinatorResponse.
        """
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000.0

        # Determine overall status
        successful = all(r.success for r in execution_results)
        status = TaskStatus.COMPLETED if successful else TaskStatus.FAILED

        # Collect results and errors
        results = [r.result for r in execution_results if r.success and r.result]
        errors = [r.error for r in execution_results if not r.success and r.error]

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(execution_results, plan)

        # Build the message
        message = self._build_message(execution_results, status)

        # Build execution metadata
        metadata = ExecutionMetadata(
            agent_id="coordinator",
            agent_type=AgentType.ORCHESTRATOR,
            started_at=start_time,
            completed_at=end_time,
            retries=sum(r.retries for r in execution_results),
        )

        return CoordinatorResponse(
            request_id=request_id,
            query=query,
            status=status,
            message=message,
            plan=plan,
            results=results,
            merged_output=merged_output,
            confidence=confidence,
            errors=errors,
            agent_results=execution_results,
            execution_metadata=metadata,
            created_at=end_time,
        )

    def _calculate_overall_confidence(
        self,
        results: list[AgentExecutionResult],
        plan: ExecutionPlan,
    ) -> Optional[ConfidenceScore]:
        """
        Calculate overall confidence from execution results.

        Args:
            results: Agent execution results.
            plan: The execution plan.

        Returns:
            A ConfidenceScore or None.
        """
        if not results:
            return None

        # Average confidence of successful results
        confidences = []
        for r in results:
            if r.success and r.result and r.result.confidence:
                confidences.append(r.result.confidence.value)

        if not confidences:
            return ConfidenceScore(
                value=0.5,
                basis="No confidence scores available from agents",
            )

        avg_confidence = sum(confidences) / len(confidences)

        # Adjust based on success rate
        success_rate = sum(1 for r in results if r.success) / len(results)
        adjusted_confidence = avg_confidence * success_rate

        return ConfidenceScore(
            value=min(1.0, adjusted_confidence),
            basis=f"Aggregated from {len(confidences)} agent confidence scores with {success_rate:.0%} success rate",
            source_module="coordinator",
        )

    def _build_message(self, results: list[AgentExecutionResult], status: TaskStatus) -> str:
        """
        Build a human-readable message from execution results.

        Args:
            results: Agent execution results.
            status: Overall status.

        Returns:
            A human-readable message string.
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if status == TaskStatus.COMPLETED:
            if len(successful) == 1:
                return f"Successfully completed task using {successful[0].agent_type.value.replace('_', ' ').title()}."
            else:
                agents = ", ".join(
                    r.agent_type.value.replace("_", " ").title()
                    for r in successful
                )
                return f"Successfully completed tasks using {len(successful)} agents: {agents}."

        elif status == TaskStatus.FAILED:
            if failed:
                errors = ", ".join(
                    f"{r.agent_type.value.replace('_', ' ').title()}: {r.error.message if r.error else 'Unknown error'}"
                    for r in failed
                )
                if successful:
                    return f"Partial completion: {len(successful)} agents succeeded, but {len(failed)} failed: {errors}"
                return f"Execution failed: {errors}"

        return "Execution completed with mixed results."

    def _create_clarification_response(
        self,
        request_id: str,
        query: str,
        plan: ExecutionPlan,
        conversation_id: Optional[str],
    ) -> CoordinatorResponse:
        """
        Create a response requesting clarification.

        Args:
            request_id: The request ID.
            query: The original query.
            plan: The execution plan.
            conversation_id: Optional conversation ID.

        Returns:
            A CoordinatorResponse with clarification request.
        """
        questions = "\n".join(f"- {q}" for q in plan.clarification_questions)

        return CoordinatorResponse(
            request_id=request_id,
            query=query,
            status=TaskStatus.WAITING_ON_DEPENDENCY,
            message=f"I need clarification before I can proceed:\n{questions}",
            plan=plan,
            warnings=["Clarification needed"],
            created_at=datetime.now(timezone.utc),
        )

    def _create_error_response(
        self,
        request_id: str,
        query: str,
        error: Exception,
        start_time: datetime,
        conversation_id: Optional[str],
    ) -> CoordinatorResponse:
        """
        Create an error response.

        Args:
            request_id: The request ID.
            query: The original query.
            error: The exception that occurred.
            start_time: When execution started.
            conversation_id: Optional conversation ID.

        Returns:
            A CoordinatorResponse with error information.
        """
        end_time = datetime.now(timezone.utc)

        return CoordinatorResponse(
            request_id=request_id,
            query=query,
            status=TaskStatus.FAILED,
            message=f"Processing failed: {str(error)}",
            errors=[
                ErrorInformation(
                    error_type=type(error).__name__,
                    message=str(error),
                    agent_type=AgentType.ORCHESTRATOR,
                )
            ],
            execution_metadata=ExecutionMetadata(
                agent_id="coordinator",
                agent_type=AgentType.ORCHESTRATOR,
                started_at=start_time,
                completed_at=end_time,
            ),
            created_at=end_time,
        )

    def get_agent_status(self) -> dict[str, Any]:
        """
        Get the status of all registered agents.

        Returns:
            A dictionary with agent status information.
        """
        status = {
            "initialized": self._initialized,
            "agents": {},
        }

        for agent_type in self.registry.list_agents():
            try:
                instance = self.registry.get_instance(agent_type)
                status["agents"][agent_type.value] = {
                    "initialized": instance.is_initialized,
                    "agent_id": instance.agent_id,
                }
            except AgentNotFoundError:
                status["agents"][agent_type.value] = {
                    "initialized": False,
                    "error": "Agent not found",
                }

        return status

    async def __aenter__(self) -> "Coordinator":
        """Support async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Support async context manager."""
        await self.shutdown()


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_coordinator(
    config: Optional[CoordinatorConfig] = None,
    registry: Optional[AgentRegistry] = None,
    memory: Optional[MemoryManager] = None,
) -> Coordinator:
    """
    Factory function to create a Coordinator instance.

    Args:
        config: Coordinator configuration.
        registry: Agent registry.
        memory: Shared MemoryManager.

    Returns:
        A configured Coordinator instance.
    """
    return Coordinator(
        config=config,
        registry=registry,
        memory=memory,
    )


__all__: list[str] = [
    "CoordinatorError",
    "CoordinatorInitializationError",
    "CoordinatorExecutionError",
    "AgentNotFoundError",
    "AgentTimeoutError",
    "CoordinatorConfig",
    "AgentExecutionResult",
    "CoordinatorResponse",
    "AgentRegistry",
    "Coordinator",
    "create_coordinator",
    "default_agent_factory",
]