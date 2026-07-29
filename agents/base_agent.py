"""
agents/base_agent.py

Abstract base class for every agent in the BioNexus Multi-Agent AI System.

This module defines the common contract, lifecycle, and request-handling
scaffolding shared by all concrete agents (``LiteratureAgent``,
``DrugInfoAgent``, ``KnowledgeGraphAgent``, ``SmilesAgent``,
``DrugDiscoveryAgent``, ``CoordinatorAgent``, and any future agent).

Design notes
------------
- This module contains ONLY the abstract agent contract, its supporting
  exception hierarchy, and configuration/lifecycle scaffolding. It does
  not implement any agent's domain behavior. Concrete agents wrap exactly
  one BioNexus module (or coordinate several, in the case of
  ``CoordinatorAgent``) and must not re-implement logic that already
  exists in Literature Search, Drug Info, Knowledge Graph, Embeddings,
  SMILES Analyzer, or Drug Discovery.
- ``BaseAgent`` is deliberately transport- and backend-agnostic. It knows
  nothing about HTTP, MCP, or any specific LLM provider; those concerns
  belong to adapters built on top of this class. This keeps the base
  class compatible with local LLMs, remote APIs, RAG pipelines, and MCP
  tool servers alike, per the Dependency Inversion Principle already
  used throughout BioNexus.
- The class exposes four abstract hooks that every concrete agent must
  implement: ``initialize``, ``validate_request``, ``execute``, and
  ``shutdown``. All lifecycle bookkeeping (state tracking, timing,
  logging, error wrapping) is handled once, here, so concrete agents only
  need to implement their domain-specific behavior.
- All public entry points return the models defined in ``agents.models``
  (``AgentResponse``, ``AgentResult``) so that callers — the
  orchestrator, an MCP tool wrapper, or a future API layer — interact
  with a single, stable contract regardless of which agent handled the
  request.
- Async-first: ``initialize``, ``execute``, and ``shutdown`` are
  coroutines so that agents can perform non-blocking I/O (calling
  external APIs, local LLM inference, database/vector-store lookups)
  without blocking the orchestrator's event loop.

Compatibility
-------------
Targets Python 3.11 and follows the same architectural conventions used
by the existing BioNexus modules (Literature Search, Drug Info,
Knowledge Graph, SMILES Analyzer, Embeddings, Drug Discovery), and
consumes ``agents.models`` exclusively for its data shapes.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Optional

from agents.models import (
    AgentRequest,
    AgentResponse,
    AgentResult,
    AgentTask,
    AgentType,
    ConfidenceScore,
    ErrorInformation,
    ExecutionMetadata,
    Priority,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-specific exception hierarchy
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """
    Base class for all exceptions raised by BioNexus agents.

    Concrete agents should prefer raising one of the subclasses below so
    that ``BaseAgent`` can translate the failure into a well-formed
    ``ErrorInformation`` instance without losing diagnostic context.

    Attributes:
        agent_type: The role of the agent that raised the error, if known
            at the point of construction.
        source_module: Name of the underlying BioNexus module that raised
            the original exception, if the failure originated downstream
            of the agent layer (e.g. "toxicity_prediction").
        is_recoverable: Whether the orchestrator may reasonably retry the
            operation that raised this error.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_type: Optional[AgentType] = None,
        source_module: Optional[str] = None,
        is_recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.agent_type = agent_type
        self.source_module = source_module
        self.is_recoverable = is_recoverable

    def to_error_information(self) -> ErrorInformation:
        """
        Convert this exception into a structured ``ErrorInformation``
        record suitable for embedding in an ``AgentResult`` or
        ``AgentResponse``.

        Returns:
            An ``ErrorInformation`` instance describing this error.
        """
        return ErrorInformation(
            error_type=type(self).__name__,
            message=self.message,
            agent_type=self.agent_type,
            source_module=self.source_module,
            is_recoverable=self.is_recoverable,
        )


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize (e.g. a dependent module
    or resource could not be set up)."""


class AgentNotInitializedError(AgentError):
    """Raised when a request is handled before the agent has completed
    initialization (``start()`` was never called, or ``stop()`` already
    ran)."""


class AgentValidationError(AgentError):
    """Raised when an ``AgentRequest`` or derived ``AgentTask`` fails
    structural or semantic validation for a specific agent."""


class AgentExecutionError(AgentError):
    """Raised when an agent's core ``execute`` logic fails, including
    failures propagated up from an underlying BioNexus module."""


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent's execution exceeds its allotted time
    budget."""


class AgentShutdownError(AgentError):
    """Raised when an agent fails to shut down or release its resources
    cleanly."""


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """
    Runtime configuration governing an agent's behavior.

    ``AgentConfig`` is intentionally generic and backend-agnostic so it
    can configure agents backed by local LLMs, remote APIs, or pure
    deterministic logic (e.g. ``SmilesAgent`` wrapping
    ``smiles_analyzer``) without a shared assumption about *how* the
    agent does its work.

    Attributes:
        default_timeout_seconds: Default wall-clock budget for a single
            ``execute`` call when the originating ``AgentRequest`` does
            not specify its own ``timeout_seconds``.
        max_retries: Maximum number of automatic re-executions the
            orchestrator (or the agent itself, for internal sub-steps)
            may attempt after a recoverable failure.
        model_name: Identifier of the LLM or predictive backend this
            agent uses, if any (e.g. a local model name or API model
            identifier). ``None`` for agents with no model backend.
        enable_rag: Whether this agent should augment its reasoning with
            retrieval-augmented generation over the Embeddings module.
        enable_tool_use: Whether this agent may invoke external tools or
            MCP endpoints during execution.
        allowed_tools: Names of tools/MCP endpoints this agent is
            permitted to invoke, when ``enable_tool_use`` is True. An
            empty list with ``enable_tool_use=True`` means "no
            restriction" and is left to the concrete agent to interpret.
        log_level: Logging level to apply to this agent's dedicated
            logger (e.g. ``logging.INFO``).
        extra: Arbitrary additional configuration not otherwise modeled,
            allowing concrete agents to accept bespoke settings without
            requiring changes to this shared dataclass.
    """

    default_timeout_seconds: float = 60.0
    max_retries: int = 2
    model_name: Optional[str] = None
    enable_rag: bool = False
    enable_tool_use: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    log_level: int = logging.INFO
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base class defining the common contract for every BioNexus
    agent.

    ``BaseAgent`` owns the parts of an agent's behavior that are
    identical across the whole fleet — identity, configuration, lifecycle
    state, request validation orchestration, timing/telemetry capture,
    and error-to-response translation — so that concrete subclasses can
    focus exclusively on the domain logic of coordinating their
    underlying BioNexus module(s).

    Subclasses must implement:
        - ``initialize``: acquire resources / warm up dependent modules.
        - ``validate_request``: agent-specific request validation.
        - ``execute``: perform the actual work for a single ``AgentTask``.
        - ``shutdown``: release resources acquired during ``initialize``.

    Subclasses should NOT override ``handle_request``, ``start``, or
    ``stop`` unless they have a specific reason to change the shared
    lifecycle/telemetry behavior — doing so risks losing the consistent
    error handling and metadata capture provided here.

    Attributes:
        agent_id: Unique identifier for this agent instance, distinct
            from ``agent_type`` (its role). Multiple instances of the
            same ``agent_type`` may run concurrently (e.g. for
            horizontal scaling) and are distinguished by ``agent_id``.
        agent_type: The role this agent fulfills within the multi-agent
            system.
        config: Runtime configuration for this agent instance.
    """

    def __init__(
        self,
        agent_type: AgentType,
        *,
        agent_id: Optional[str] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        """
        Initialize shared agent state. Subclasses should call
        ``super().__init__(...)`` before performing any of their own
        constructor logic, and should perform actual resource acquisition
        in ``initialize`` rather than here.

        Args:
            agent_type: The role this agent fulfills.
            agent_id: Unique identifier for this agent instance. A UUID4
                string is generated if not supplied.
            config: Runtime configuration for this agent instance. A
                default ``AgentConfig`` is used if not supplied.
        """
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.agent_type: AgentType = agent_type
        self.config: AgentConfig = config or AgentConfig()

        self._initialized: bool = False
        self._initialized_at: Optional[datetime] = None

        self._logger: logging.Logger = logging.getLogger(
            f"{__name__}.{self.agent_type.value}.{self.agent_id[:8]}"
        )
        self._logger.setLevel(self.config.log_level)

    # -- Identity / state -----------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Return whether this agent has completed ``start()`` and has
        not since been shut down via ``stop()``."""
        return self._initialized

    @property
    def logger(self) -> logging.Logger:
        """Return this agent's dedicated logger."""
        return self._logger

    def get_capabilities(self) -> list[str]:
        """
        Return a list of capability identifiers this agent supports.

        Intended for discovery by the orchestrator or an MCP tool
        registry (e.g. to decide whether this agent can fulfill a given
        ``AgentRequest``). The default implementation returns an empty
        list; concrete agents are expected to override this with their
        actual capabilities (e.g. ``["compound_similarity_search",
        "target_prediction"]``).

        Returns:
            A list of capability identifier strings.
        """
        return []

    # -- Abstract lifecycle hooks -----------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """
        Perform agent-specific startup work.

        Called exactly once by ``start()`` before this agent handles any
        requests. Implementations should acquire whatever resources they
        need — e.g. opening a connection managed by an underlying
        BioNexus module, warming a local LLM, or validating that a
        required API key is configured — and raise
        ``AgentInitializationError`` on failure rather than allowing a
        raw exception to propagate.

        Raises:
            AgentInitializationError: If initialization fails.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_request(self, request: AgentRequest) -> None:
        """
        Validate that ``request`` is well-formed and executable by this
        agent.

        Called by ``handle_request`` before any task is constructed or
        executed. Implementations should check whatever is specific to
        this agent (e.g. that ``request.parameters`` contains a valid
        SMILES string for ``SmilesAgent``, or a compound identifier for
        ``DrugInfoAgent``) and raise ``AgentValidationError`` with a
        descriptive message when the request cannot be handled. A
        request that passes validation should return normally (this
        method has no meaningful return value).

        Args:
            request: The inbound request to validate.

        Raises:
            AgentValidationError: If the request is invalid for this
                agent.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Perform the actual work for a single ``AgentTask``.

        Implementations must coordinate the appropriate underlying
        BioNexus module(s) to fulfill ``task`` — they must not
        re-implement logic that belongs in Literature Search, Drug Info,
        Knowledge Graph, Embeddings, SMILES Analyzer, or Drug Discovery.
        Implementations should raise ``AgentExecutionError`` (or a
        subclass, such as ``AgentTimeoutError``) on failure rather than
        allowing an unrelated exception type to propagate; ``handle_request``
        will translate any exception into a failed ``AgentResult``
        regardless, but a project-specific exception carries richer,
        more actionable diagnostic information.

        Args:
            task: The task to execute.

        Returns:
            An ``AgentResult`` describing the outcome. Implementations
            are free to construct this directly, but should generally
            prefer the ``_build_success_result`` / ``_build_error_result``
            helpers on this base class for consistency.

        Raises:
            AgentExecutionError: If execution fails.
        """
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Perform agent-specific teardown work.

        Called exactly once by ``stop()``. Implementations should release
        whatever resources were acquired in ``initialize`` (close
        connections, flush caches, cancel background work) and raise
        ``AgentShutdownError`` on failure rather than allowing a raw
        exception to propagate.

        Raises:
            AgentShutdownError: If shutdown fails.
        """
        raise NotImplementedError

    # -- Lifecycle orchestration (concrete) -------------------------------

    async def start(self) -> None:
        """
        Bring this agent online.

        Invokes the ``initialize`` hook, records lifecycle state, and
        logs the outcome. Safe to call only once per agent instance;
        calling it again while already initialized is a no-op that logs
        a warning rather than re-running ``initialize``.

        Raises:
            AgentInitializationError: If ``initialize`` fails. The
                agent remains in the "not initialized" state.
        """
        if self._initialized:
            self._logger.warning(
                "start() called on agent '%s' (%s) that is already initialized; ignoring.",
                self.agent_id,
                self.agent_type.value,
            )
            return

        self._logger.info(
            "Initializing agent '%s' (%s)...", self.agent_id, self.agent_type.value
        )
        try:
            await self.initialize()
        except AgentError:
            self._logger.exception(
                "Agent '%s' (%s) failed to initialize.",
                self.agent_id,
                self.agent_type.value,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - deliberate translation boundary
            self._logger.exception(
                "Agent '%s' (%s) failed to initialize due to an unexpected error.",
                self.agent_id,
                self.agent_type.value,
            )
            raise AgentInitializationError(
                f"Unexpected error during initialization: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc

        self._initialized = True
        self._initialized_at = datetime.now(timezone.utc)
        self._logger.info(
            "Agent '%s' (%s) initialized successfully.",
            self.agent_id,
            self.agent_type.value,
        )

    async def stop(self) -> None:
        """
        Take this agent offline.

        Invokes the ``shutdown`` hook and records lifecycle state. Safe
        to call even if the agent was never successfully initialized;
        in that case ``shutdown`` is still invoked to allow best-effort
        cleanup of any partially acquired resources.

        Raises:
            AgentShutdownError: If ``shutdown`` fails.
        """
        self._logger.info(
            "Shutting down agent '%s' (%s)...", self.agent_id, self.agent_type.value
        )
        try:
            await self.shutdown()
        except AgentError:
            self._logger.exception(
                "Agent '%s' (%s) failed to shut down cleanly.",
                self.agent_id,
                self.agent_type.value,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - deliberate translation boundary
            self._logger.exception(
                "Agent '%s' (%s) failed to shut down due to an unexpected error.",
                self.agent_id,
                self.agent_type.value,
            )
            raise AgentShutdownError(
                f"Unexpected error during shutdown: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            ) from exc
        finally:
            self._initialized = False

        self._logger.info(
            "Agent '%s' (%s) shut down successfully.",
            self.agent_id,
            self.agent_type.value,
        )

    async def __aenter__(self) -> "BaseAgent":
        """Support ``async with SomeAgent(...) as agent:`` usage."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Ensure the agent is shut down when leaving an ``async with``
        block, even if the block raised."""
        await self.stop()

    # -- Request handling (concrete) --------------------------------------

    async def handle_request(self, request: AgentRequest) -> AgentResponse:
        """
        Validate, execute, and respond to a single ``AgentRequest``.

        This is the primary entry point orchestration code should call.
        It performs, in order: an initialization check, request
        validation, task construction, timed execution via ``execute``,
        and translation of the outcome (success or failure) into a
        well-formed ``AgentResponse``. Any exception raised by
        ``validate_request`` or ``execute`` — whether a project-specific
        ``AgentError`` or an unexpected exception — is caught here and
        converted into a failed ``AgentResponse`` rather than propagating
        to the caller, so orchestration code never needs its own
        per-agent try/except handling.

        Args:
            request: The inbound request to fulfill.

        Returns:
            An ``AgentResponse`` describing the outcome. Its ``status``
            is ``TaskStatus.COMPLETED`` on success or ``TaskStatus.FAILED``
            (or ``TaskStatus.TIMED_OUT``) on failure; it never raises.
        """
        execution_metadata = ExecutionMetadata(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model_used=self.config.model_name,
        )

        if not self._initialized:
            error = AgentNotInitializedError(
                f"Agent '{self.agent_id}' ({self.agent_type.value}) "
                "received a request before start() was called.",
                agent_type=self.agent_type,
                is_recoverable=True,
            )
            return self._build_error_response(request, error, execution_metadata)

        try:
            self.validate_request(request)
        except AgentError as error:
            self._logger.warning(
                "Request '%s' failed validation for agent '%s': %s",
                request.request_id,
                self.agent_id,
                error.message,
            )
            return self._build_error_response(request, error, execution_metadata)
        except Exception as exc:  # noqa: BLE001 - deliberate translation boundary
            self._logger.exception(
                "Unexpected error validating request '%s' for agent '%s'.",
                request.request_id,
                self.agent_id,
            )
            error = AgentValidationError(
                f"Unexpected validation error: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            )
            return self._build_error_response(request, error, execution_metadata)

        task = self._build_task(request)

        try:
            result = await self.execute(task)
        except AgentError as error:
            self._logger.error(
                "Execution failed for task '%s' (agent '%s'): %s",
                task.task_id,
                self.agent_id,
                error.message,
            )
            return self._build_error_response(
                request, error, execution_metadata, task=task
            )
        except Exception as exc:  # noqa: BLE001 - deliberate translation boundary
            self._logger.exception(
                "Unexpected error executing task '%s' (agent '%s').",
                task.task_id,
                self.agent_id,
            )
            error = AgentExecutionError(
                f"Unexpected execution error: {exc}",
                agent_type=self.agent_type,
                is_recoverable=False,
            )
            return self._build_error_response(
                request, error, execution_metadata, task=task
            )
        finally:
            execution_metadata.completed_at = datetime.now(timezone.utc)

        self._logger.info(
            "Request '%s' fulfilled by agent '%s' in %.2f ms (task '%s', status=%s).",
            request.request_id,
            self.agent_id,
            execution_metadata.duration_ms or 0.0,
            task.task_id,
            result.status.value,
        )

        return AgentResponse(
            request_id=request.request_id,
            status=result.status,
            message=self._summarize_result(result),
            results=[result],
            confidence=result.confidence,
            errors=[result.error] if result.error is not None else [],
            execution_metadata=execution_metadata,
        )

    # -- Result / response construction helpers ---------------------------

    def _build_task(self, request: AgentRequest) -> AgentTask:
        """
        Construct an ``AgentTask`` for this agent from an ``AgentRequest``.

        Args:
            request: The originating request.

        Returns:
            A new ``AgentTask`` bound to this agent's ``agent_type`` and
            assigned to this agent instance.
        """
        # Capture a single timestamp for both `created_at` and `started_at`
        # so construction-time validation (started_at >= created_at) cannot
        # fail due to the sub-millisecond gap between two separate
        # `datetime.now()` calls.
        now = datetime.now(timezone.utc)
        return AgentTask(
            request_id=request.request_id,
            agent_type=self.agent_type,
            description=request.instruction,
            input_data=dict(request.parameters),
            status=TaskStatus.RUNNING,
            priority=request.priority,
            assigned_agent_id=self.agent_id,
            created_at=now,
            started_at=now,
        )

    def _build_success_result(
        self,
        task: AgentTask,
        output: Any,
        *,
        confidence: Optional[ConfidenceScore] = None,
        execution_metadata: Optional[ExecutionMetadata] = None,
    ) -> AgentResult:
        """
        Construct a successful ``AgentResult`` for ``task``.

        Concrete agents should call this helper at the end of a
        successful ``execute`` implementation rather than constructing
        ``AgentResult`` directly, to keep result construction consistent
        across the agent fleet.

        Args:
            task: The task this result fulfills.
            output: The agent's output payload (an existing BioNexus
                domain model or a collection thereof).
            confidence: Confidence in the output, if applicable.
            execution_metadata: Telemetry for this specific execution. If
                omitted, a minimal ``ExecutionMetadata`` is constructed
                using ``task.started_at`` (or now, if unset) as the start
                time and the current time as completion.

        Returns:
            A completed ``AgentResult``.
        """
        metadata = execution_metadata or ExecutionMetadata(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            started_at=task.started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            model_used=self.config.model_name,
        )
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=TaskStatus.COMPLETED,
            output=output,
            confidence=confidence,
            execution_metadata=metadata,
        )

    def _build_error_result(
        self,
        task: AgentTask,
        error: AgentError,
        *,
        status: TaskStatus = TaskStatus.FAILED,
        execution_metadata: Optional[ExecutionMetadata] = None,
    ) -> AgentResult:
        """
        Construct a failed ``AgentResult`` for ``task``.

        Args:
            task: The task this result reports on.
            error: The project-specific exception describing the failure.
            status: The terminal status to report; must be either
                ``TaskStatus.FAILED`` or ``TaskStatus.TIMED_OUT``.
            execution_metadata: Telemetry for this specific execution. If
                omitted, a minimal ``ExecutionMetadata`` is constructed
                using ``task.started_at`` (or now, if unset) as the start
                time and the current time as completion.

        Returns:
            A failed (or timed-out) ``AgentResult``.

        Raises:
            ValueError: If ``status`` is not a failure-representing
                status.
        """
        if status not in (TaskStatus.FAILED, TaskStatus.TIMED_OUT):
            raise ValueError(
                "status must be TaskStatus.FAILED or TaskStatus.TIMED_OUT."
            )
        metadata = execution_metadata or ExecutionMetadata(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            started_at=task.started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            model_used=self.config.model_name,
        )
        return AgentResult(
            task_id=task.task_id,
            agent_type=self.agent_type,
            status=status,
            output=None,
            execution_metadata=metadata,
            error=error.to_error_information(),
        )

    def _build_error_response(
        self,
        request: AgentRequest,
        error: AgentError,
        execution_metadata: ExecutionMetadata,
        *,
        task: Optional[AgentTask] = None,
    ) -> AgentResponse:
        """
        Construct a failed ``AgentResponse`` for ``request``.

        Args:
            request: The request that could not be fulfilled.
            error: The project-specific exception describing the failure.
            execution_metadata: Top-level telemetry for this request.
            task: The task that was being executed when the failure
                occurred, if one had already been constructed. When
                provided, a corresponding failed ``AgentResult`` is
                included in the response's ``results``.

        Returns:
            An ``AgentResponse`` with ``status`` set to
            ``TaskStatus.FAILED`` (or ``TaskStatus.TIMED_OUT`` for
            ``AgentTimeoutError``) and populated ``errors``.
        """
        execution_metadata.completed_at = (
            execution_metadata.completed_at or datetime.now(timezone.utc)
        )
        failure_status = (
            TaskStatus.TIMED_OUT
            if isinstance(error, AgentTimeoutError)
            else TaskStatus.FAILED
        )
        results: list[AgentResult] = []
        if task is not None:
            results.append(
                self._build_error_result(
                    task,
                    error,
                    status=failure_status,
                    execution_metadata=execution_metadata,
                )
            )
        return AgentResponse(
            request_id=request.request_id,
            status=failure_status,
            message=f"Request failed: {error.message}",
            results=results,
            errors=[error.to_error_information()],
            execution_metadata=execution_metadata,
        )

    @staticmethod
    def _summarize_result(result: AgentResult) -> str:
        """
        Produce a short human-readable summary line for a completed
        ``AgentResult``.

        Concrete agents are free to override how richer natural-language
        summaries are produced (e.g. via a local LLM synthesis step); this
        default keeps ``handle_request`` self-contained and dependency-free.

        Args:
            result: The result to summarize.

        Returns:
            A short status message suitable for ``AgentResponse.message``.
        """
        if result.is_success:
            confidence_note = (
                f" (confidence: {result.confidence.label})"
                if result.confidence is not None
                else ""
            )
            return f"Task '{result.task_id}' completed successfully{confidence_note}."
        reason = result.error.message if result.error is not None else "unknown error"
        return f"Task '{result.task_id}' did not complete successfully: {reason}"

    def __repr__(self) -> str:
        """Return an unambiguous representation useful for logging/debugging."""
        return (
            f"{type(self).__name__}(agent_id={self.agent_id!r}, "
            f"agent_type={self.agent_type.value!r}, "
            f"initialized={self._initialized})"
        )


__all__: list[str] = [
    "AgentError",
    "AgentInitializationError",
    "AgentNotInitializedError",
    "AgentValidationError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AgentShutdownError",
    "AgentConfig",
    "BaseAgent",
]