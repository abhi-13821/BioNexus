"""
agents/models.py

Data models for the BioNexus Multi-Agent AI System.

This module defines the core data structures used to orchestrate and
coordinate the existing BioNexus modules (Literature Search, Drug Info,
Knowledge Graph, Embeddings, SMILES Analyzer, and Drug Discovery) through
a multi-agent architecture.

Design notes
------------
- This module contains ONLY data models (dataclasses), enums, and
  configuration containers. No orchestration logic, agent implementations,
  routing, or I/O operations are implemented here — those live in
  ``agents/orchestrator.py``, ``agents/router.py``, and individual agent
  modules (e.g. ``agents/literature_agent.py``), which are expected to
  *consume* these models rather than duplicate the domain logic already
  implemented in the Literature Search, Drug Info, Knowledge Graph,
  Embeddings, SMILES Analyzer, and Drug Discovery modules.
- Agents never re-implement domain logic. An ``AgentTask`` targeting, for
  example, target prediction is expected to be fulfilled by an agent that
  internally calls ``drug_discovery.target_prediction``; this module only
  models the *shape* of the request/response/state passed between agents
  and the orchestrator.
- Validation is limited to structural/range checks on the data itself
  (e.g., scores must be within [0, 1]), performed in ``__post_init__``.
- Identifiers default to UUID4 strings (consistent with
  ``embeddings.models.EmbeddingRecord``) so callers never need to manage
  ID uniqueness manually.
- Timestamps are timezone-aware UTC ``datetime`` objects, consistent with
  ``drug_discovery.models.DrugDiscoveryResult``.
- Models are intentionally future-proof: ``AgentRequest``, ``AgentTask``,
  and ``ExecutionMetadata`` carry open-ended ``metadata``/``parameters``
  dictionaries so that future local LLMs, RAG retrieval steps, MCP tool
  invocations, or external API calls can be threaded through the system
  without breaking the schema. ``AgentMemoryRecord`` references embeddings
  by ``embedding_id`` (a plain string) rather than embedding the vector
  itself, keeping this module decoupled from the Embeddings module's
  storage/inference backend (Dependency Inversion Principle).

Compatibility
-------------
Targets Python 3.11 and follows the same architectural conventions used
by the existing BioNexus modules (Literature Search, Drug Info,
Knowledge Graph, SMILES Analyzer, Embeddings, Drug Discovery).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentType(str, Enum):
    """
    Classification of an agent's role within the BioNexus multi-agent
    system.

    Each non-orchestration member corresponds to a thin coordination layer
    over an existing BioNexus module; the agent itself must not
    re-implement that module's logic.
    """

    ORCHESTRATOR = "orchestrator"
    LITERATURE_SEARCH_AGENT = "literature_search_agent"
    DRUG_INFO_AGENT = "drug_info_agent"
    KNOWLEDGE_GRAPH_AGENT = "knowledge_graph_agent"
    SMILES_ANALYSIS_AGENT = "smiles_analysis_agent"
    EMBEDDING_AGENT = "embedding_agent"
    COMPOUND_SIMILARITY_AGENT = "compound_similarity_agent"
    TARGET_PREDICTION_AGENT = "target_prediction_agent"
    TOXICITY_PREDICTION_AGENT = "toxicity_prediction_agent"
    DRUG_REPURPOSING_AGENT = "drug_repurposing_agent"
    RAG_AGENT = "rag_agent"
    TOOL_AGENT = "tool_agent"
    GENERAL_QA_AGENT = "general_qa_agent"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """Lifecycle status of an ``AgentTask`` or ``WorkflowState``."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_ON_DEPENDENCY = "waiting_on_dependency"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Return whether this status represents a final state."""
        return self in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        )


class Priority(IntEnum):
    """
    Relative execution priority for requests and tasks.

    Implemented as an ``IntEnum`` (rather than a plain string enum) so
    that priorities can be directly compared and used as sort keys by
    scheduling/orchestration code, e.g. ``sorted(tasks, key=lambda t:
    -t.priority)``.
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Supporting value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceScore:
    """
    A normalized confidence measurement attached to an agent's output.

    Attributes:
        value: Numeric confidence in [0.0, 1.0].
        basis: Free-text explanation of what the score is derived from
            (e.g. "aggregated from 3 target prediction hits",
            "LLM self-reported confidence", "retrieval similarity score").
        source_module: Name of the underlying BioNexus module or model
            that produced the signal this score is based on, if
            applicable (e.g. "target_prediction", "embeddings").
    """

    value: float
    basis: Optional[str] = None
    source_module: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the confidence value range."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"value must be within [0.0, 1.0], got {self.value}."
            )

    @property
    def label(self) -> str:
        """
        Return a qualitative bucket for this confidence score.

        Returns:
            One of "low", "medium", "high", "very_high".
        """
        if self.value < 0.4:
            return "low"
        if self.value < 0.7:
            return "medium"
        if self.value < 0.9:
            return "high"
        return "very_high"


@dataclass
class ExecutionMetadata:
    """
    Telemetry describing how an agent's task execution was carried out.

    Attributes:
        agent_id: Unique identifier of the agent instance that performed
            the execution (distinct from ``AgentType``, which identifies
            the agent's *role*).
        agent_type: The role of the executing agent.
        started_at: UTC timestamp when execution began.
        completed_at: UTC timestamp when execution finished, if complete.
        model_used: Identifier of the LLM or predictive model invoked
            during execution, if any (e.g. a local LLM name, or a
            BioNexus model/method identifier).
        tools_invoked: Names of tools or MCP endpoints invoked during
            execution, in call order (e.g. ["compound_similarity.search",
            "target_prediction.predict"]).
        tokens_used: Total LLM token usage for this execution, if
            applicable.
        retries: Number of retry attempts consumed before this execution
            concluded.
        host_environment: Free-text description of where execution ran
            (e.g. "local", "gpu-worker-1", "external_api").
        extra: Arbitrary additional telemetry not otherwise modeled.
    """

    agent_id: str
    agent_type: AgentType
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    model_used: Optional[str] = None
    tools_invoked: list[str] = field(default_factory=list)
    tokens_used: Optional[int] = None
    retries: int = 0
    host_environment: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers, timestamp ordering, and counters."""
        if not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string.")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at.")
        if self.retries < 0:
            raise ValueError("retries cannot be negative.")
        if self.tokens_used is not None and self.tokens_used < 0:
            raise ValueError("tokens_used cannot be negative.")

    @property
    def duration_ms(self) -> Optional[float]:
        """
        Return the execution duration in milliseconds.

        Returns:
            Elapsed time between ``started_at`` and ``completed_at`` in
            milliseconds, or None if execution has not yet completed.
        """
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000.0


@dataclass
class ErrorInformation:
    """
    Structured error detail attached to a failed ``AgentResult`` or
    ``AgentResponse``.

    Attributes:
        error_type: Short machine-readable error category (e.g.
            "ToolInvocationError", "Timeout", "ValidationError",
            "UpstreamModuleError").
        message: Human-readable error message.
        agent_type: The role of the agent that raised or observed the
            error, if applicable.
        source_module: Name of the underlying BioNexus module that raised
            the original exception, if the error originated downstream
            of the agent layer (e.g. "toxicity_prediction").
        is_recoverable: Whether the orchestrator may reasonably retry the
            associated task after this error.
        occurred_at: UTC timestamp when the error was recorded.
        traceback_snippet: Optional truncated traceback or diagnostic
            text for debugging. Should not contain sensitive data.
    """

    error_type: str
    message: str
    agent_type: Optional[AgentType] = None
    source_module: Optional[str] = None
    is_recoverable: bool = False
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    traceback_snippet: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate required identifying fields."""
        if not self.error_type.strip():
            raise ValueError("error_type must be a non-empty string.")
        if not self.message.strip():
            raise ValueError("message must be a non-empty string.")


# ---------------------------------------------------------------------------
# Request / task models
# ---------------------------------------------------------------------------


@dataclass
class AgentRequest:
    """
    An inbound request for the multi-agent system to fulfill.

    An ``AgentRequest`` represents the top-level unit of work submitted
    by a caller (a user, another agent, or an external API/MCP client).
    The orchestrator decomposes a request into one or more ``AgentTask``
    instances, each routed to the appropriate agent.

    Attributes:
        request_id: Unique identifier for this request.
        instruction: Natural-language or structured instruction describing
            what the caller wants accomplished (e.g. "find drug
            candidates similar to aspirin with low predicted toxicity").
        agent_type: Preferred or explicitly requested agent to handle this
            request, if the caller (or a routing layer) has already
            determined it. May be ``None`` to let the orchestrator decide.
        parameters: Structured parameters supporting the instruction (e.g.
            a SMILES string, a compound ID, a similarity threshold).
            Downstream agents are responsible for validating these against
            the parameter schema of the module they wrap.
        conversation_id: Identifier of the ``ConversationContext`` this
            request belongs to, if part of an ongoing conversation.
        parent_request_id: Identifier of a parent ``AgentRequest`` if this
            request was spawned as a sub-request by another agent.
        priority: Relative scheduling priority for this request.
        requested_by: Identifier of the requesting principal (user ID,
            agent ID, or external API client), if known.
        timeout_seconds: Maximum wall-clock time allotted to fulfill this
            request before it is considered timed out.
        created_at: UTC timestamp when the request was created.
        metadata: Arbitrary additional attributes not otherwise modeled
            (e.g. RAG retrieval hints, MCP tool allowlist, client locale).
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str = ""
    agent_type: Optional[AgentType] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    parent_request_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    requested_by: Optional[str] = None
    timeout_seconds: Optional[float] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and instruction content."""
        if not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string.")
        if not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided.")


@dataclass
class AgentTask:
    """
    A single unit of work assigned to a specific agent, typically produced
    by decomposing an ``AgentRequest``.

    Attributes:
        task_id: Unique identifier for this task.
        request_id: Identifier of the originating ``AgentRequest``.
        agent_type: The agent role responsible for executing this task.
        description: Human-readable description of what the task does
            (e.g. "predict binding targets for compound C001").
        input_data: Structured input the assigned agent needs to execute
            the task (e.g. a SMILES string, a list of compound IDs).
        dependencies: Identifiers of other ``AgentTask`` instances that
            must reach a terminal ``TaskStatus`` before this task may
            start (used to express DAG-style workflows).
        status: Current lifecycle status of the task.
        priority: Relative scheduling priority, typically inherited from
            the originating request but independently overridable.
        assigned_agent_id: Identifier of the concrete agent instance
            assigned to execute this task, once assigned.
        created_at: UTC timestamp when the task was created.
        started_at: UTC timestamp when execution began, if started.
        completed_at: UTC timestamp when the task reached a terminal
            status, if finished.
        metadata: Arbitrary additional attributes not otherwise modeled.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    agent_type: AgentType = AgentType.ORCHESTRATOR
    description: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.NORMAL
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and timestamp ordering."""
        if not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string.")
        if not self.description.strip():
            raise ValueError("description must be a non-empty string.")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("started_at cannot be earlier than created_at.")
        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at cannot be earlier than started_at.")

    def is_ready(self, completed_task_ids: set[str]) -> bool:
        """
        Return whether this task's dependencies have all completed.

        Args:
            completed_task_ids: Identifiers of tasks that have already
                reached a terminal, successful status.

        Returns:
            True if every entry in ``dependencies`` is present in
            ``completed_task_ids``, False otherwise.
        """
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)


# ---------------------------------------------------------------------------
# Result / response models
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """
    The outcome produced by an agent executing a single ``AgentTask``.

    Attributes:
        result_id: Unique identifier for this result.
        task_id: Identifier of the ``AgentTask`` this result fulfills.
        agent_type: The role of the agent that produced this result.
        status: Terminal status of the underlying task execution.
        output: The agent's output payload. Expected to hold instances of
            (or serialized references to) existing BioNexus domain models
            — e.g. a list of ``DrugCandidate``, a ``ToxicityPrediction``,
            or a list of ``SimilaritySearchResult`` — rather than
            re-derived data.
        confidence: Aggregate confidence in the output, if applicable.
        execution_metadata: Telemetry describing how the result was
            produced.
        error: Structured error detail, populated when ``status`` is
            ``TaskStatus.FAILED`` or ``TaskStatus.TIMED_OUT``.
        created_at: UTC timestamp when this result was recorded.
    """

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_type: AgentType = AgentType.ORCHESTRATOR
    status: TaskStatus = TaskStatus.COMPLETED
    output: Any = None
    confidence: Optional[ConfidenceScore] = None
    execution_metadata: Optional[ExecutionMetadata] = None
    error: Optional[ErrorInformation] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        """Validate identifiers and status/error consistency."""
        if not self.result_id.strip():
            raise ValueError("result_id must be a non-empty string.")
        if not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string.")
        if self.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT) and self.error is None:
            raise ValueError(
                "error must be provided when status is FAILED or TIMED_OUT."
            )

    @property
    def is_success(self) -> bool:
        """Return whether this result represents a successful completion."""
        return self.status == TaskStatus.COMPLETED and self.error is None


@dataclass
class AgentResponse:
    """
    The top-level response returned to the caller for an ``AgentRequest``,
    aggregating one or more ``AgentResult`` instances.

    Attributes:
        response_id: Unique identifier for this response.
        request_id: Identifier of the originating ``AgentRequest``.
        status: Overall status summarizing all constituent results (e.g.
            ``TaskStatus.COMPLETED`` if every result succeeded,
            ``TaskStatus.FAILED`` if any critical result failed).
        message: Human-readable summary of the outcome, suitable for
            direct display to an end user (e.g. a natural-language
            synthesis produced by an LLM over the underlying results).
        results: The individual ``AgentResult`` instances that make up
            this response.
        confidence: Aggregate confidence across all results, if
            applicable.
        errors: Structured errors collected from any failed results.
        execution_metadata: Top-level telemetry for the overall request
            (e.g. orchestrator-level timing), distinct from the
            per-result ``ExecutionMetadata`` values nested in ``results``.
        created_at: UTC timestamp when this response was finalized.
        metadata: Arbitrary additional attributes not otherwise modeled.
    """

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    status: TaskStatus = TaskStatus.COMPLETED
    message: str = ""
    results: list[AgentResult] = field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    errors: list[ErrorInformation] = field(default_factory=list)
    execution_metadata: Optional[ExecutionMetadata] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers."""
        if not self.response_id.strip():
            raise ValueError("response_id must be a non-empty string.")
        if not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string.")

    def successful_results(self) -> list[AgentResult]:
        """
        Return the subset of ``results`` that completed successfully.

        Returns:
            A list of ``AgentResult`` instances where ``is_success`` is
            True.
        """
        return [r for r in self.results if r.is_success]

    def failed_results(self) -> list[AgentResult]:
        """
        Return the subset of ``results`` that did not complete
        successfully.

        Returns:
            A list of ``AgentResult`` instances where ``is_success`` is
            False.
        """
        return [r for r in self.results if not r.is_success]


# ---------------------------------------------------------------------------
# Memory / conversation / workflow models
# ---------------------------------------------------------------------------


@dataclass
class AgentMemoryRecord:
    """
    A single turn or fact stored in an agent's short- or long-term memory.

    This model deliberately references embeddings by identifier rather
    than embedding vectors directly. Vector generation, storage, and
    retrieval remain the responsibility of the Embeddings module
    (``embeddings.embedding_generator``, ``embeddings.embedding_store``,
    ``embeddings.embedding_search``); this module only records the link.

    Attributes:
        memory_id: Unique identifier for this memory record.
        conversation_id: Identifier of the owning ``ConversationContext``.
        agent_type: The agent role that authored or observed this memory
            entry, if applicable (``None`` for user-authored entries).
        role: Origin of the content, e.g. "user", "agent", "system",
            "tool".
        content: The text content of this memory entry.
        embedding_id: Identifier of a corresponding
            ``embeddings.models.EmbeddingRecord`` for this content, if one
            has been generated and stored, enabling semantic recall.
        created_at: UTC timestamp when this memory entry was created.
        metadata: Arbitrary additional attributes not otherwise modeled
            (e.g. referenced compound/target IDs, tool call details).
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    agent_type: Optional[AgentType] = None
    role: str = "user"
    content: str = ""
    embedding_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and content."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string.")
        if not self.content.strip():
            raise ValueError("content must be a non-empty string.")
        if not self.role.strip():
            raise ValueError("role must be a non-empty string.")


@dataclass
class ConversationContext:
    """
    The accumulated context of an ongoing multi-turn interaction with the
    multi-agent system.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        user_id: Identifier of the human or system principal driving the
            conversation, if known.
        active_workflow_id: Identifier of the ``WorkflowState`` currently
            executing on behalf of this conversation, if any.
        history: Ordered list of ``AgentMemoryRecord`` entries comprising
            this conversation's turn-by-turn history.
        created_at: UTC timestamp when the conversation began.
        updated_at: UTC timestamp of the most recent activity in this
            conversation.
        metadata: Arbitrary additional attributes not otherwise modeled
            (e.g. session locale, client channel, feature flags).
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    active_workflow_id: Optional[str] = None
    history: list[AgentMemoryRecord] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and timestamp ordering."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string.")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at.")

    def recent_history(self, limit: int = 10) -> list[AgentMemoryRecord]:
        """
        Return the most recent memory entries in this conversation.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            Up to ``limit`` of the most recent ``AgentMemoryRecord``
            entries, ordered oldest to newest.

        Raises:
            ValueError: If ``limit`` is not a positive integer.
        """
        if limit < 1:
            raise ValueError("limit must be a positive integer.")
        return self.history[-limit:]


@dataclass
class WorkflowState:
    """
    The execution state of a multi-step, potentially multi-agent workflow
    spawned to fulfill one or more ``AgentRequest`` instances.

    A ``WorkflowState`` tracks a DAG of ``AgentTask`` instances (linked via
    ``AgentTask.dependencies``) and the ``AgentResult`` instances produced
    as tasks complete, allowing the orchestrator to resume, inspect, or
    cancel long-running or asynchronous workflows.

    Attributes:
        workflow_id: Unique identifier for this workflow.
        conversation_id: Identifier of the owning ``ConversationContext``,
            if this workflow was spawned within a conversation.
        request_ids: Identifiers of the ``AgentRequest`` instances this
            workflow is fulfilling.
        tasks: All ``AgentTask`` instances that make up this workflow.
        results: ``AgentResult`` instances produced so far for tasks in
            ``tasks``.
        status: Overall status of the workflow.
        current_stage: Free-text label identifying the current stage of
            execution (e.g. "similarity_search", "target_prediction"),
            for progress reporting.
        created_at: UTC timestamp when the workflow was created.
        updated_at: UTC timestamp of the most recent state change.
        completed_at: UTC timestamp when the workflow reached a terminal
            status, if finished.
        metadata: Arbitrary additional attributes not otherwise modeled.
    """

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    request_ids: list[str] = field(default_factory=list)
    tasks: list[AgentTask] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    current_stage: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and timestamp ordering."""
        if not self.workflow_id.strip():
            raise ValueError("workflow_id must be a non-empty string.")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be earlier than created_at.")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot be earlier than created_at.")

    def is_completed(self) -> bool:
        """
        Return whether this workflow has reached a terminal status.

        Returns:
            True if ``status`` is a terminal ``TaskStatus``, False
            otherwise.
        """
        return self.status.is_terminal

    def pending_tasks(self) -> list[AgentTask]:
        """
        Return tasks that have not yet reached a terminal status.

        Returns:
            A list of ``AgentTask`` instances whose ``status`` is not
            terminal.
        """
        return [t for t in self.tasks if not t.status.is_terminal]

    def failed_tasks(self) -> list[AgentTask]:
        """
        Return tasks that failed or timed out.

        Returns:
            A list of ``AgentTask`` instances with status
            ``TaskStatus.FAILED`` or ``TaskStatus.TIMED_OUT``.
        """
        return [
            t
            for t in self.tasks
            if t.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT)
        ]

    def ready_tasks(self) -> list[AgentTask]:
        """
        Return pending tasks whose dependencies have all completed.

        Returns:
            A list of ``AgentTask`` instances with status
            ``TaskStatus.PENDING`` that are ready to be scheduled, based
            on the set of tasks with status ``TaskStatus.COMPLETED``.
        """
        completed_ids = {
            t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING and t.is_ready(completed_ids)
        ]

    def progress_percentage(self) -> float:
        """
        Return the fraction of tasks that have reached a terminal status.

        Returns:
            A percentage in [0.0, 100.0]. Returns 0.0 for a workflow with
            no tasks.
        """
        if not self.tasks:
            return 0.0
        terminal_count = sum(1 for t in self.tasks if t.status.is_terminal)
        return (terminal_count / len(self.tasks)) * 100.0


__all__: list[str] = [
    "AgentType",
    "TaskStatus",
    "Priority",
    "ConfidenceScore",
    "ExecutionMetadata",
    "ErrorInformation",
    "AgentRequest",
    "AgentTask",
    "AgentResult",
    "AgentResponse",
    "AgentMemoryRecord",
    "ConversationContext",
    "WorkflowState",
]