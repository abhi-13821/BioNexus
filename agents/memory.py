"""
agents/memory.py

Memory subsystem for the BioNexus Multi-Agent AI System.

This module gives agents (subclasses of ``agents.base_agent.BaseAgent``)
and the orchestrator a place to persist and retrieve conversation
history, completed tasks, agent outputs, and workflow state across a
single run or an ongoing multi-turn interaction.

Design notes
------------
- This module contains ONLY memory storage/retrieval concerns. It does
  not implement any agent's domain behavior, and it does not depend on
  frontend modules (``literature_search``, ``drug_info``,
  ``knowledge_graphs``, ``smiles_analyzer``) — it depends only on
  ``agents.models`` and the standard library.
- The storage contract is defined by the abstract ``MemoryStore`` class.
  ``InMemoryMemoryStore`` is the concrete short-term implementation
  provided here: fast, process-local, thread-safe, and bounded (with
  optional TTL/size-based eviction) so it is safe to use as a rolling
  working memory. A future long-term backend (e.g. one backed by a
  database or by ``embeddings.embedding_store``) can be dropped in by
  implementing the same ``MemoryStore`` interface — nothing above the
  storage layer needs to change.
- ``MemoryManager`` is the ergonomic facade agents actually hold a
  reference to. It composes a ``MemoryStore`` (defaulting to
  ``InMemoryMemoryStore``) and exposes agent-friendly convenience
  methods (e.g. ``remember_turn``, ``record_task_result``,
  ``checkpoint_workflow``) so that call sites in ``BaseAgent`` subclasses
  stay short and declarative.
- Vector/semantic memory is explicitly out of scope for this module's
  own logic (that belongs to the Embeddings module), but it is a
  first-class extension point: ``VectorMemoryBackend`` defines the
  contract a future embeddings-backed long-term memory must satisfy
  (built on ``embeddings.embedding_generator`` /
  ``embeddings.embedding_search``), and ``AgentMemoryRecord.embedding_id``
  (defined in ``agents.models``) is the join key connecting a memory
  record to its vector representation once one exists.
  ``MemoryManager.semantic_search`` delegates to an injected
  ``VectorMemoryBackend`` when one is configured, and raises a clear,
  actionable ``AgentMemoryError`` otherwise.
- Concurrency: ``InMemoryMemoryStore`` guards all reads and writes with a
  single re-entrant lock (``threading.RLock``). This is deliberately
  simple (favoring correctness over throughput) since memory operations
  are in-memory dict/list manipulations and are not expected to be a
  bottleneck; it also makes the store safe to share across agent
  instances running on separate threads (e.g. via
  ``asyncio.to_thread``/an executor), not just within a single event
  loop.
- Isolation: records are deep-copied on write and on read so that
  callers mutating a returned object cannot corrupt the store's internal
  state, and so that the store cannot be corrupted by later mutation of
  an object a caller passed in.
- Serialization: ``to_serializable`` recursively converts the dataclasses
  in ``agents.models`` (plus enums and datetimes) into JSON-safe
  primitives, and ``MemoryStore`` exposes ``export_*`` helpers built on
  top of it. Reconstructing a store from serialized data is intentionally
  left to a future persistent backend, since that requires the schema
  awareness a real persistence layer (e.g. a database ORM) already has.

Compatibility
-------------
Targets Python 3.11. Consumes ``agents.models`` exclusively for its data
shapes and is designed to be held by ``BaseAgent`` subclasses (e.g. as
``self.memory: MemoryManager``) without base_agent.py needing to import
this module.
"""

from __future__ import annotations

import copy
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Optional

from agents.models import (
    AgentMemoryRecord,
    AgentResult,
    AgentTask,
    AgentType,
    ConversationContext,
    TaskStatus,
    WorkflowState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory-specific exception hierarchy
# ---------------------------------------------------------------------------


class AgentMemoryError(Exception):
    """Base class for all exceptions raised by the memory subsystem."""


class MemoryNotFoundError(AgentMemoryError):
    """Raised when a requested memory record does not exist."""


class MemoryValidationError(AgentMemoryError):
    """Raised when a memory operation is given structurally invalid
    arguments (e.g. an unknown field passed to an update)."""


class MemoryCapacityError(AgentMemoryError):
    """Raised when a bounded memory collection is full and the caller
    must explicitly free space before adding more records (used where
    silent eviction would be too destructive, e.g. whole conversations)."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MemoryRecordType(str, Enum):
    """Classification of a record stored in a ``MemoryStore``, used to
    scope search and maintenance operations across heterogeneous storage
    collections."""

    CONVERSATION_MESSAGE = "conversation_message"
    TASK = "task"
    RESULT = "result"
    WORKFLOW = "workflow"


class MemoryScope(str, Enum):
    """
    Intended retention scope for a memory record.

    ``InMemoryMemoryStore`` only ever holds ``SHORT_TERM`` data (it is
    process-local and unbounded persistence is not its job), but records
    and stores are tagged with scope so that a future composite store
    (short-term + long-term) can decide what to promote and what to
    let expire.
    """

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MemoryConfig:
    """
    Bounds and retention policy for an ``InMemoryMemoryStore``.

    Attributes:
        max_messages_per_conversation: Maximum number of
            ``AgentMemoryRecord`` entries retained per conversation. When
            exceeded, the oldest messages are evicted (FIFO) so the
            conversation stays a bounded rolling window. ``None`` means
            unbounded.
        message_ttl_seconds: Maximum age, in seconds, a conversation
            message may reach before it becomes eligible for pruning.
            ``None`` disables TTL-based pruning.
        max_conversations: Maximum number of distinct conversations the
            store will hold concurrently. Unlike message eviction,
            exceeding this raises ``MemoryCapacityError`` rather than
            silently discarding a whole conversation's history — callers
            must explicitly ``delete_conversation`` first. ``None`` means
            unbounded.
        max_tasks: Maximum number of ``AgentTask`` records retained.
            When exceeded, the oldest (by ``created_at``) task is evicted.
            ``None`` means unbounded.
        max_results: Maximum number of ``AgentResult`` records retained.
            When exceeded, the oldest (by ``created_at``) result is
            evicted. ``None`` means unbounded.
        max_workflows: Maximum number of ``WorkflowState`` records
            retained. When exceeded, the oldest completed workflow (by
            ``created_at``) is evicted in preference to an active one.
            ``None`` means unbounded.
        auto_prune_on_write: Whether to opportunistically prune expired
            conversation messages every time a message is appended (in
            addition to the manual ``prune_expired`` entry point).
    """

    max_messages_per_conversation: Optional[int] = 200
    message_ttl_seconds: Optional[float] = None
    max_conversations: Optional[int] = 1000
    max_tasks: Optional[int] = 5000
    max_results: Optional[int] = 5000
    max_workflows: Optional[int] = 1000
    auto_prune_on_write: bool = True

    def __post_init__(self) -> None:
        """Validate that configured bounds are positive when provided."""
        for name in (
            "max_messages_per_conversation",
            "max_conversations",
            "max_tasks",
            "max_results",
            "max_workflows",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer when set.")
        if self.message_ttl_seconds is not None and self.message_ttl_seconds <= 0:
            raise ValueError("message_ttl_seconds must be positive when set.")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def to_serializable(value: Any) -> Any:
    """
    Recursively convert a value into JSON-safe primitives.

    Handles the dataclasses defined in ``agents.models`` (and any nested
    dataclasses), ``Enum`` members (converted to their ``.value``),
    ``datetime`` instances (converted via ``isoformat``), and standard
    containers. Any other value is returned unchanged, so callers should
    only rely on the result being JSON-safe when the input is built
    exclusively from the types above.

    Args:
        value: The value to convert.

    Returns:
        A structure of ``dict``, ``list``, ``str``, ``int``, ``float``,
        ``bool``, and ``None`` suitable for ``json.dumps``.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_serializable(getattr(value, f.name)) for f in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Search result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemorySearchResult:
    """
    A single scored match returned from ``MemoryStore.search``.

    Attributes:
        record_type: The kind of record matched.
        record_id: Identifier of the matched record (``memory_id``,
            ``task_id``, ``result_id``, or ``workflow_id`` depending on
            ``record_type``).
        score: Relevance score in [0.0, 1.0]; higher is more relevant.
            Computed via simple keyword overlap in
            ``InMemoryMemoryStore`` — a future vector-backed store may
            populate this from cosine similarity instead.
        snippet: A short excerpt of the matched text, for display in
            search results without dereferencing the full record.
        record: The matched record itself (an ``AgentMemoryRecord``,
            ``AgentTask``, ``AgentResult``, or ``WorkflowState``).
    """

    record_type: MemoryRecordType
    record_id: str
    score: float
    snippet: str
    record: Any

    def __post_init__(self) -> None:
        """Validate the relevance score range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within [0.0, 1.0], got {self.score}.")


# ---------------------------------------------------------------------------
# Vector memory extension point (compatibility hook, not implemented here)
# ---------------------------------------------------------------------------


class VectorMemoryBackend(ABC):
    """
    Contract for a future embeddings-backed long-term/semantic memory.

    This module does not implement a vector backend — that belongs to
    the Embeddings module (``embeddings.embedding_generator``,
    ``embeddings.embedding_store``, ``embeddings.embedding_search``).
    A concrete implementation there can wrap those pieces and be handed
    to ``MemoryManager`` to enable ``semantic_search`` without any change
    to this module.
    """

    @abstractmethod
    def index_record(self, record: AgentMemoryRecord) -> str:
        """
        Generate and persist an embedding for ``record``.

        Args:
            record: The memory record to index.

        Returns:
            The identifier of the resulting embedding (suitable for
            storing in ``record.embedding_id``), matching the
            ``embedding_id`` scheme used by
            ``embeddings.models.EmbeddingRecord``.
        """
        raise NotImplementedError

    @abstractmethod
    def semantic_search(
        self, query: str, *, top_k: int = 5
    ) -> list[tuple[AgentMemoryRecord, float]]:
        """
        Retrieve the memory records most semantically similar to
        ``query``.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(record, score)`` pairs ordered by descending
            similarity score.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MemoryStore contract
# ---------------------------------------------------------------------------


class MemoryStore(ABC):
    """
    Abstract contract for storing and retrieving multi-agent memory.

    Implementations own conversation history, completed tasks, agent
    outputs (results), and workflow state, and must provide search,
    update, delete, and serialization support for each. ``InMemoryMemoryStore``
    is the concrete short-term implementation provided in this module;
    a future long-term implementation (persistent database, or
    vector-augmented store) should implement this same interface so that
    ``MemoryManager`` and agent code do not need to change.
    """

    # -- Conversations -----------------------------------------------------

    @abstractmethod
    def create_conversation(
        self, *, conversation_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> ConversationContext:
        """Create and store a new, empty conversation."""
        raise NotImplementedError

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> ConversationContext:
        """Retrieve a stored conversation by identifier."""
        raise NotImplementedError

    @abstractmethod
    def conversation_exists(self, conversation_id: str) -> bool:
        """Return whether a conversation with this identifier is stored."""
        raise NotImplementedError

    @abstractmethod
    def append_message(
        self, conversation_id: str, record: AgentMemoryRecord
    ) -> ConversationContext:
        """Append a memory record to an existing conversation's history."""
        raise NotImplementedError

    @abstractmethod
    def update_message(
        self, conversation_id: str, memory_id: str, **changes: Any
    ) -> AgentMemoryRecord:
        """Update fields of a specific message within a conversation."""
        raise NotImplementedError

    @abstractmethod
    def delete_message(self, conversation_id: str, memory_id: str) -> None:
        """Remove a specific message from a conversation's history."""
        raise NotImplementedError

    @abstractmethod
    def get_recent_messages(
        self, conversation_id: str, limit: int = 10
    ) -> list[AgentMemoryRecord]:
        """Retrieve the most recent messages in a conversation."""
        raise NotImplementedError

    @abstractmethod
    def list_conversations(self) -> list[str]:
        """Return the identifiers of all stored conversations."""
        raise NotImplementedError

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and its entire history."""
        raise NotImplementedError

    # -- Tasks ---------------------------------------------------------------

    @abstractmethod
    def save_task(self, task: AgentTask) -> AgentTask:
        """Create or overwrite a stored task."""
        raise NotImplementedError

    @abstractmethod
    def get_task(self, task_id: str) -> AgentTask:
        """Retrieve a stored task by identifier."""
        raise NotImplementedError

    @abstractmethod
    def update_task(self, task_id: str, **changes: Any) -> AgentTask:
        """Update fields of an existing task."""
        raise NotImplementedError

    @abstractmethod
    def delete_task(self, task_id: str) -> None:
        """Delete a stored task."""
        raise NotImplementedError

    @abstractmethod
    def list_tasks(
        self,
        *,
        request_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        agent_type: Optional[AgentType] = None,
    ) -> list[AgentTask]:
        """List stored tasks, optionally filtered."""
        raise NotImplementedError

    # -- Results / agent outputs ---------------------------------------------

    @abstractmethod
    def save_result(self, result: AgentResult) -> AgentResult:
        """Create or overwrite a stored agent result."""
        raise NotImplementedError

    @abstractmethod
    def get_result(self, result_id: str) -> AgentResult:
        """Retrieve a stored result by identifier."""
        raise NotImplementedError

    @abstractmethod
    def delete_result(self, result_id: str) -> None:
        """Delete a stored result."""
        raise NotImplementedError

    @abstractmethod
    def list_results(
        self,
        *,
        task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> list[AgentResult]:
        """List stored results, optionally filtered."""
        raise NotImplementedError

    # -- Workflows -------------------------------------------------------

    @abstractmethod
    def save_workflow(self, workflow: WorkflowState) -> WorkflowState:
        """Create or overwrite a stored workflow state."""
        raise NotImplementedError

    @abstractmethod
    def get_workflow(self, workflow_id: str) -> WorkflowState:
        """Retrieve a stored workflow state by identifier."""
        raise NotImplementedError

    @abstractmethod
    def update_workflow(self, workflow_id: str, **changes: Any) -> WorkflowState:
        """Update fields of an existing workflow state."""
        raise NotImplementedError

    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> None:
        """Delete a stored workflow state."""
        raise NotImplementedError

    @abstractmethod
    def list_workflows(
        self,
        *,
        conversation_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> list[WorkflowState]:
        """List stored workflow states, optionally filtered."""
        raise NotImplementedError

    # -- Search / maintenance ------------------------------------------------

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        record_types: Optional[Iterable[MemoryRecordType]] = None,
        conversation_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """Search stored records for a free-text query."""
        raise NotImplementedError

    @abstractmethod
    def prune_expired(self) -> int:
        """Evict records that have exceeded their configured TTL."""
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Return record counts per collection, for monitoring/debugging."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from the store."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory (short-term) implementation
# ---------------------------------------------------------------------------


class InMemoryMemoryStore(MemoryStore):
    """
    Thread-safe, process-local, short-term implementation of
    ``MemoryStore``.

    Suitable as the default working memory for a running BioNexus
    process: fast, dependency-free, and bounded by ``MemoryConfig`` so it
    cannot grow without limit over a long session. Not durable — process
    restart loses all state. A persistent or vector-augmented backend
    should implement ``MemoryStore`` directly rather than subclass this
    implementation.

    All public methods are safe to call concurrently from multiple
    threads; a single re-entrant lock guards the store's internal state.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        """
        Initialize an empty store.

        Args:
            config: Bounds and retention policy for this store. A
                default ``MemoryConfig`` is used if not supplied.
        """
        self.config: MemoryConfig = config or MemoryConfig()
        self._lock = threading.RLock()

        self._conversations: dict[str, ConversationContext] = {}
        self._tasks: dict[str, AgentTask] = {}
        self._results: dict[str, AgentResult] = {}
        self._workflows: dict[str, WorkflowState] = {}

        self._logger = logging.getLogger(f"{__name__}.InMemoryMemoryStore")

    # -- Conversations -----------------------------------------------------

    def create_conversation(
        self, *, conversation_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> ConversationContext:
        """
        Create and store a new, empty conversation.

        Args:
            conversation_id: Identifier to assign to the new conversation.
                A UUID4 string is generated if not supplied.
            user_id: Identifier of the principal driving the conversation.

        Returns:
            The newly created ``ConversationContext``.

        Raises:
            MemoryValidationError: If ``conversation_id`` is already in
                use.
            MemoryCapacityError: If ``config.max_conversations`` would be
                exceeded.
        """
        with self._lock:
            kwargs: dict[str, Any] = {}
            if conversation_id is not None:
                kwargs["conversation_id"] = conversation_id
            if user_id is not None:
                kwargs["user_id"] = user_id
            context = ConversationContext(**kwargs)

            if context.conversation_id in self._conversations:
                raise MemoryValidationError(
                    f"Conversation '{context.conversation_id}' already exists."
                )
            if (
                self.config.max_conversations is not None
                and len(self._conversations) >= self.config.max_conversations
            ):
                raise MemoryCapacityError(
                    "Maximum number of concurrent conversations "
                    f"({self.config.max_conversations}) reached; delete an "
                    "existing conversation before creating another."
                )

            stored = copy.deepcopy(context)
            self._conversations[context.conversation_id] = stored
            self._logger.debug("Created conversation '%s'.", context.conversation_id)
            return copy.deepcopy(stored)

    def get_conversation(self, conversation_id: str) -> ConversationContext:
        """
        Retrieve a stored conversation by identifier.

        Args:
            conversation_id: Identifier of the conversation to retrieve.

        Returns:
            A defensive copy of the stored ``ConversationContext``.

        Raises:
            MemoryNotFoundError: If no such conversation is stored.
        """
        with self._lock:
            context = self._conversations.get(conversation_id)
            if context is None:
                raise MemoryNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )
            return replace(context, history=list(context.history))

    def conversation_exists(self, conversation_id: str) -> bool:
        """Return whether a conversation with this identifier is stored."""
        with self._lock:
            return conversation_id in self._conversations

    def append_message(
        self, conversation_id: str, record: AgentMemoryRecord
    ) -> ConversationContext:
        """
        Append a memory record to an existing conversation's history.

        If ``record.conversation_id`` does not match ``conversation_id``,
        a corrected copy of ``record`` is stored instead (the identifier
        of the containing conversation is authoritative). Applies FIFO
        eviction and, if configured, TTL-based pruning after the append
        so the conversation stays within ``config.max_messages_per_conversation``.

        Args:
            conversation_id: Identifier of the conversation to append to.
            record: The memory record to append.

        Returns:
            The updated ``ConversationContext``.

        Raises:
            MemoryNotFoundError: If no such conversation is stored.
        """
        with self._lock:
            context = self._conversations.get(conversation_id)
            if context is None:
                raise MemoryNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )

            if record.conversation_id != conversation_id:
                record = replace(record, conversation_id=conversation_id)

            history = list(context.history)
            history.append(record)

            max_messages = self.config.max_messages_per_conversation
            if max_messages is not None and len(history) > max_messages:
                evicted = len(history) - max_messages
                history = history[evicted:]
                self._logger.debug(
                    "Evicted %d oldest message(s) from conversation '%s' "
                    "(limit=%d).",
                    evicted,
                    conversation_id,
                    max_messages,
                )

            updated = replace(
                context,
                history=history,
                updated_at=datetime.now(timezone.utc),
            )
            self._conversations[conversation_id] = updated

            if self.config.auto_prune_on_write and self.config.message_ttl_seconds:
                self._prune_conversation_messages(conversation_id)
                updated = self._conversations[conversation_id]

            return replace(updated, history=list(updated.history))

    def update_message(
        self, conversation_id: str, memory_id: str, **changes: Any
    ) -> AgentMemoryRecord:
        """
        Update fields of a specific message within a conversation.

        Args:
            conversation_id: Identifier of the containing conversation.
            memory_id: Identifier of the message to update.
            **changes: Field names and new values to apply, as accepted
                by ``dataclasses.replace`` on ``AgentMemoryRecord``.

        Returns:
            The updated ``AgentMemoryRecord``.

        Raises:
            MemoryNotFoundError: If the conversation or message does not
                exist.
            MemoryValidationError: If ``changes`` references an unknown
                field or produces an invalid record.
        """
        with self._lock:
            context = self._conversations.get(conversation_id)
            if context is None:
                raise MemoryNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )
            history = list(context.history)
            for index, existing in enumerate(history):
                if existing.memory_id == memory_id:
                    try:
                        updated_record = replace(existing, **changes)
                    except TypeError as exc:
                        raise MemoryValidationError(
                            f"Invalid update for message '{memory_id}': {exc}"
                        ) from exc
                    history[index] = updated_record
                    self._conversations[conversation_id] = replace(
                        context,
                        history=history,
                        updated_at=datetime.now(timezone.utc),
                    )
                    return updated_record
            raise MemoryNotFoundError(
                f"Message '{memory_id}' was not found in conversation "
                f"'{conversation_id}'."
            )

    def delete_message(self, conversation_id: str, memory_id: str) -> None:
        """
        Remove a specific message from a conversation's history.

        Args:
            conversation_id: Identifier of the containing conversation.
            memory_id: Identifier of the message to remove.

        Raises:
            MemoryNotFoundError: If the conversation or message does not
                exist.
        """
        with self._lock:
            context = self._conversations.get(conversation_id)
            if context is None:
                raise MemoryNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )
            history = [m for m in context.history if m.memory_id != memory_id]
            if len(history) == len(context.history):
                raise MemoryNotFoundError(
                    f"Message '{memory_id}' was not found in conversation "
                    f"'{conversation_id}'."
                )
            self._conversations[conversation_id] = replace(
                context, history=history, updated_at=datetime.now(timezone.utc)
            )

    def get_recent_messages(
        self, conversation_id: str, limit: int = 10
    ) -> list[AgentMemoryRecord]:
        """
        Retrieve the most recent messages in a conversation.

        Args:
            conversation_id: Identifier of the conversation.
            limit: Maximum number of messages to return.

        Returns:
            Up to ``limit`` of the most recent ``AgentMemoryRecord``
            entries, ordered oldest to newest.

        Raises:
            MemoryNotFoundError: If no such conversation is stored.
        """
        context = self.get_conversation(conversation_id)
        return context.recent_history(limit=limit)

    def list_conversations(self) -> list[str]:
        """Return the identifiers of all stored conversations."""
        with self._lock:
            return list(self._conversations.keys())

    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation and its entire history.

        Args:
            conversation_id: Identifier of the conversation to delete.

        Raises:
            MemoryNotFoundError: If no such conversation is stored.
        """
        with self._lock:
            if conversation_id not in self._conversations:
                raise MemoryNotFoundError(
                    f"Conversation '{conversation_id}' was not found."
                )
            del self._conversations[conversation_id]
            self._logger.debug("Deleted conversation '%s'.", conversation_id)

    # -- Tasks ---------------------------------------------------------------

    def save_task(self, task: AgentTask) -> AgentTask:
        """
        Create or overwrite a stored task.

        Applies FIFO eviction of the oldest stored task (by
        ``created_at``) if ``config.max_tasks`` would otherwise be
        exceeded by adding a genuinely new task.

        Args:
            task: The task to store.

        Returns:
            A defensive copy of the stored task.
        """
        with self._lock:
            is_new = task.task_id not in self._tasks
            if (
                is_new
                and self.config.max_tasks is not None
                and len(self._tasks) >= self.config.max_tasks
            ):
                self._evict_oldest(self._tasks, key=lambda t: t.created_at)
            stored = replace(task)
            self._tasks[task.task_id] = stored
            return replace(stored)

    def get_task(self, task_id: str) -> AgentTask:
        """
        Retrieve a stored task by identifier.

        Args:
            task_id: Identifier of the task to retrieve.

        Returns:
            A defensive copy of the stored ``AgentTask``.

        Raises:
            MemoryNotFoundError: If no such task is stored.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise MemoryNotFoundError(f"Task '{task_id}' was not found.")
            return replace(task)

    def update_task(self, task_id: str, **changes: Any) -> AgentTask:
        """
        Update fields of an existing task.

        Args:
            task_id: Identifier of the task to update.
            **changes: Field names and new values to apply, as accepted
                by ``dataclasses.replace`` on ``AgentTask``.

        Returns:
            The updated ``AgentTask``.

        Raises:
            MemoryNotFoundError: If no such task is stored.
            MemoryValidationError: If ``changes`` references an unknown
                field or produces an invalid task.
        """
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing is None:
                raise MemoryNotFoundError(f"Task '{task_id}' was not found.")
            try:
                updated = replace(existing, **changes)
            except TypeError as exc:
                raise MemoryValidationError(
                    f"Invalid update for task '{task_id}': {exc}"
                ) from exc
            self._tasks[task_id] = updated
            return replace(updated)

    def delete_task(self, task_id: str) -> None:
        """
        Delete a stored task.

        Args:
            task_id: Identifier of the task to delete.

        Raises:
            MemoryNotFoundError: If no such task is stored.
        """
        with self._lock:
            if task_id not in self._tasks:
                raise MemoryNotFoundError(f"Task '{task_id}' was not found.")
            del self._tasks[task_id]

    def list_tasks(
        self,
        *,
        request_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        agent_type: Optional[AgentType] = None,
    ) -> list[AgentTask]:
        """
        List stored tasks, optionally filtered.

        Args:
            request_id: If provided, only tasks with this
                ``request_id`` are returned.
            status: If provided, only tasks with this ``status`` are
                returned.
            agent_type: If provided, only tasks with this ``agent_type``
                are returned.

        Returns:
            Matching ``AgentTask`` instances, ordered by ``created_at``
            ascending.
        """
        with self._lock:
            tasks = list(self._tasks.values())
        if request_id is not None:
            tasks = [t for t in tasks if t.request_id == request_id]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if agent_type is not None:
            tasks = [t for t in tasks if t.agent_type == agent_type]
        tasks.sort(key=lambda t: t.created_at)
        return [replace(t) for t in tasks]

    # -- Results / agent outputs ---------------------------------------------

    def save_result(self, result: AgentResult) -> AgentResult:
        """
        Create or overwrite a stored agent result.

        Applies FIFO eviction of the oldest stored result (by
        ``created_at``) if ``config.max_results`` would otherwise be
        exceeded by adding a genuinely new result.

        Args:
            result: The result to store.

        Returns:
            A defensive copy of the stored result.
        """
        with self._lock:
            is_new = result.result_id not in self._results
            if (
                is_new
                and self.config.max_results is not None
                and len(self._results) >= self.config.max_results
            ):
                self._evict_oldest(self._results, key=lambda r: r.created_at)
            stored = replace(result)
            self._results[result.result_id] = stored
            return replace(stored)

    def get_result(self, result_id: str) -> AgentResult:
        """
        Retrieve a stored result by identifier.

        Args:
            result_id: Identifier of the result to retrieve.

        Returns:
            A defensive copy of the stored ``AgentResult``.

        Raises:
            MemoryNotFoundError: If no such result is stored.
        """
        with self._lock:
            result = self._results.get(result_id)
            if result is None:
                raise MemoryNotFoundError(f"Result '{result_id}' was not found.")
            return replace(result)

    def delete_result(self, result_id: str) -> None:
        """
        Delete a stored result.

        Args:
            result_id: Identifier of the result to delete.

        Raises:
            MemoryNotFoundError: If no such result is stored.
        """
        with self._lock:
            if result_id not in self._results:
                raise MemoryNotFoundError(f"Result '{result_id}' was not found.")
            del self._results[result_id]

    def list_results(
        self,
        *,
        task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> list[AgentResult]:
        """
        List stored results, optionally filtered.

        Args:
            task_id: If provided, only results for this ``task_id`` are
                returned.
            status: If provided, only results with this ``status`` are
                returned.

        Returns:
            Matching ``AgentResult`` instances, ordered by ``created_at``
            ascending.
        """
        with self._lock:
            results = list(self._results.values())
        if task_id is not None:
            results = [r for r in results if r.task_id == task_id]
        if status is not None:
            results = [r for r in results if r.status == status]
        results.sort(key=lambda r: r.created_at)
        return [replace(r) for r in results]

    # -- Workflows -------------------------------------------------------

    def save_workflow(self, workflow: WorkflowState) -> WorkflowState:
        """
        Create or overwrite a stored workflow state.

        Applies eviction of the oldest completed workflow (preferring to
        evict a terminal-status workflow over an active one) if
        ``config.max_workflows`` would otherwise be exceeded by adding a
        genuinely new workflow.

        Args:
            workflow: The workflow state to store.

        Returns:
            A defensive copy of the stored workflow state.

        Raises:
            MemoryCapacityError: If the limit is reached and no
                terminal-status workflow is available to evict.
        """
        with self._lock:
            is_new = workflow.workflow_id not in self._workflows
            if (
                is_new
                and self.config.max_workflows is not None
                and len(self._workflows) >= self.config.max_workflows
            ):
                self._evict_oldest_workflow()
            stored = replace(workflow)
            self._workflows[workflow.workflow_id] = stored
            return replace(stored)

    def get_workflow(self, workflow_id: str) -> WorkflowState:
        """
        Retrieve a stored workflow state by identifier.

        Args:
            workflow_id: Identifier of the workflow to retrieve.

        Returns:
            A defensive copy of the stored ``WorkflowState``.

        Raises:
            MemoryNotFoundError: If no such workflow is stored.
        """
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None:
                raise MemoryNotFoundError(f"Workflow '{workflow_id}' was not found.")
            return replace(workflow)

    def update_workflow(self, workflow_id: str, **changes: Any) -> WorkflowState:
        """
        Update fields of an existing workflow state.

        Args:
            workflow_id: Identifier of the workflow to update.
            **changes: Field names and new values to apply, as accepted
                by ``dataclasses.replace`` on ``WorkflowState``.

        Returns:
            The updated ``WorkflowState``.

        Raises:
            MemoryNotFoundError: If no such workflow is stored.
            MemoryValidationError: If ``changes`` references an unknown
                field or produces an invalid workflow state.
        """
        with self._lock:
            existing = self._workflows.get(workflow_id)
            if existing is None:
                raise MemoryNotFoundError(f"Workflow '{workflow_id}' was not found.")
            changes.setdefault("updated_at", datetime.now(timezone.utc))
            try:
                updated = replace(existing, **changes)
            except TypeError as exc:
                raise MemoryValidationError(
                    f"Invalid update for workflow '{workflow_id}': {exc}"
                ) from exc
            self._workflows[workflow_id] = updated
            return replace(updated)

    def delete_workflow(self, workflow_id: str) -> None:
        """
        Delete a stored workflow state.

        Args:
            workflow_id: Identifier of the workflow to delete.

        Raises:
            MemoryNotFoundError: If no such workflow is stored.
        """
        with self._lock:
            if workflow_id not in self._workflows:
                raise MemoryNotFoundError(f"Workflow '{workflow_id}' was not found.")
            del self._workflows[workflow_id]

    def list_workflows(
        self,
        *,
        conversation_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> list[WorkflowState]:
        """
        List stored workflow states, optionally filtered.

        Args:
            conversation_id: If provided, only workflows with this
                ``conversation_id`` are returned.
            status: If provided, only workflows with this ``status`` are
                returned.

        Returns:
            Matching ``WorkflowState`` instances, ordered by
            ``created_at`` ascending.
        """
        with self._lock:
            workflows = list(self._workflows.values())
        if conversation_id is not None:
            workflows = [w for w in workflows if w.conversation_id == conversation_id]
        if status is not None:
            workflows = [w for w in workflows if w.status == status]
        workflows.sort(key=lambda w: w.created_at)
        return [replace(w) for w in workflows]

    # -- Search / maintenance ------------------------------------------------

    def search(
        self,
        query: str,
        *,
        record_types: Optional[Iterable[MemoryRecordType]] = None,
        conversation_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """
        Search stored records for a free-text query.

        Uses simple case-insensitive keyword overlap scoring against a
        per-record-type searchable text extraction (conversation message
        content; task descriptions; result outputs and error messages;
        workflow stage and metadata). This is intentionally lightweight
        so the store has no external dependencies; a
        ``VectorMemoryBackend`` can be layered on top via
        ``MemoryManager.semantic_search`` for genuine semantic recall.

        Args:
            query: Free-text query. Matching is case-insensitive and
                based on whitespace-tokenized terms.
            record_types: If provided, restrict the search to these
                record types. Defaults to searching all types.
            conversation_id: If provided, restrict conversation-message
                results to this conversation (other record types are
                unaffected by this filter, since they are not always
                conversation-scoped).
            limit: Maximum number of results to return.

        Returns:
            Matching ``MemorySearchResult`` instances, ordered by
            descending relevance score.

        Raises:
            ValueError: If ``query`` is empty or ``limit`` is not
                positive.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string.")
        if limit < 1:
            raise ValueError("limit must be a positive integer.")

        terms = [t for t in query.lower().split() if t]
        types_to_search = set(record_types) if record_types else set(MemoryRecordType)
        matches: list[MemorySearchResult] = []

        with self._lock:
            if MemoryRecordType.CONVERSATION_MESSAGE in types_to_search:
                conversations = (
                    [self._conversations[conversation_id]]
                    if conversation_id and conversation_id in self._conversations
                    else list(self._conversations.values())
                )
                for context in conversations:
                    for message in context.history:
                        score = _keyword_score(message.content, terms)
                        if score > 0.0:
                            matches.append(
                                MemorySearchResult(
                                    record_type=MemoryRecordType.CONVERSATION_MESSAGE,
                                    record_id=message.memory_id,
                                    score=score,
                                    snippet=_snippet(message.content),
                                    record=replace(message),
                                )
                            )

            if MemoryRecordType.TASK in types_to_search:
                for task in self._tasks.values():
                    score = _keyword_score(task.description, terms)
                    if score > 0.0:
                        matches.append(
                            MemorySearchResult(
                                record_type=MemoryRecordType.TASK,
                                record_id=task.task_id,
                                score=score,
                                snippet=_snippet(task.description),
                                record=replace(task),
                            )
                        )

            if MemoryRecordType.RESULT in types_to_search:
                for result in self._results.values():
                    text = _result_searchable_text(result)
                    score = _keyword_score(text, terms)
                    if score > 0.0:
                        matches.append(
                            MemorySearchResult(
                                record_type=MemoryRecordType.RESULT,
                                record_id=result.result_id,
                                score=score,
                                snippet=_snippet(text),
                                record=replace(result),
                            )
                        )

            if MemoryRecordType.WORKFLOW in types_to_search:
                for workflow in self._workflows.values():
                    text = " ".join(
                        part
                        for part in (workflow.current_stage, str(workflow.metadata))
                        if part
                    )
                    score = _keyword_score(text, terms)
                    if score > 0.0:
                        matches.append(
                            MemorySearchResult(
                                record_type=MemoryRecordType.WORKFLOW,
                                record_id=workflow.workflow_id,
                                score=score,
                                snippet=_snippet(text),
                                record=replace(workflow),
                            )
                        )

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]

    def prune_expired(self) -> int:
        """
        Evict conversation messages that have exceeded
        ``config.message_ttl_seconds``.

        Other record collections (tasks, results, workflows) are not
        subject to TTL expiry in this implementation, since a completed
        task/result/workflow is typically meaningful for the lifetime of
        the run rather than a fixed duration; they are instead bounded by
        the FIFO/preferential eviction applied in ``save_task``,
        ``save_result``, and ``save_workflow``.

        Returns:
            The total number of messages evicted across all
            conversations.
        """
        if not self.config.message_ttl_seconds:
            return 0
        with self._lock:
            total_evicted = 0
            for conversation_id in list(self._conversations.keys()):
                total_evicted += self._prune_conversation_messages(conversation_id)
            if total_evicted:
                self._logger.debug(
                    "Pruned %d expired message(s) across %d conversation(s).",
                    total_evicted,
                    len(self._conversations),
                )
            return total_evicted

    def stats(self) -> dict[str, int]:
        """
        Return record counts per collection, for monitoring/debugging.

        Returns:
            A dict with keys ``"conversations"``, ``"messages"``,
            ``"tasks"``, ``"results"``, and ``"workflows"``.
        """
        with self._lock:
            return {
                "conversations": len(self._conversations),
                "messages": sum(
                    len(c.history) for c in self._conversations.values()
                ),
                "tasks": len(self._tasks),
                "results": len(self._results),
                "workflows": len(self._workflows),
            }

    def clear(self) -> None:
        """Remove all records from the store."""
        with self._lock:
            self._conversations.clear()
            self._tasks.clear()
            self._results.clear()
            self._workflows.clear()
            self._logger.debug("Cleared all memory records.")

    # -- Serialization ---------------------------------------------------

    def export_conversation(self, conversation_id: str) -> dict[str, Any]:
        """
        Export a conversation as a JSON-safe dict.

        Args:
            conversation_id: Identifier of the conversation to export.

        Returns:
            The conversation, recursively converted via
            ``to_serializable``.

        Raises:
            MemoryNotFoundError: If no such conversation is stored.
        """
        return to_serializable(self.get_conversation(conversation_id))

    def export_workflow(self, workflow_id: str) -> dict[str, Any]:
        """
        Export a workflow state as a JSON-safe dict.

        Args:
            workflow_id: Identifier of the workflow to export.

        Returns:
            The workflow state, recursively converted via
            ``to_serializable``.

        Raises:
            MemoryNotFoundError: If no such workflow is stored.
        """
        return to_serializable(self.get_workflow(workflow_id))

    def export_all(self) -> dict[str, Any]:
        """
        Export the entire store as a JSON-safe dict snapshot.

        Intended for diagnostics or as a starting point for a future
        persistent backend's write-through/backup logic; reconstructing
        a store from this snapshot is left to that backend.

        Returns:
            A dict with keys ``"conversations"``, ``"tasks"``,
            ``"results"``, and ``"workflows"``, each a list of
            ``to_serializable``-converted records.
        """
        with self._lock:
            return {
                "conversations": [
                    to_serializable(c) for c in self._conversations.values()
                ],
                "tasks": [to_serializable(t) for t in self._tasks.values()],
                "results": [to_serializable(r) for r in self._results.values()],
                "workflows": [to_serializable(w) for w in self._workflows.values()],
            }

    # -- Internal helpers --------------------------------------------------

    def _prune_conversation_messages(self, conversation_id: str) -> int:
        """
        Remove expired messages from a single conversation.

        Must be called while holding ``self._lock``.

        Args:
            conversation_id: Identifier of the conversation to prune.

        Returns:
            The number of messages evicted.
        """
        ttl = self.config.message_ttl_seconds
        if not ttl:
            return 0
        context = self._conversations.get(conversation_id)
        if context is None:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl)
        kept = [m for m in context.history if m.created_at >= cutoff]
        evicted = len(context.history) - len(kept)
        if evicted:
            self._conversations[conversation_id] = replace(context, history=kept)
        return evicted

    @staticmethod
    def _evict_oldest(collection: dict[str, Any], *, key: Any) -> None:
        """
        Remove the oldest entry from ``collection`` according to ``key``.

        Must be called while holding ``self._lock``.

        Args:
            collection: The dict to evict from (mutated in place).
            key: A callable extracting a sortable timestamp from a value.
        """
        if not collection:
            return
        oldest_id = min(collection, key=lambda k: key(collection[k]))
        del collection[oldest_id]

    def _evict_oldest_workflow(self) -> None:
        """
        Evict the oldest workflow, preferring a terminal-status workflow
        over an active one so in-progress work is not silently lost.

        Must be called while holding ``self._lock``.
        """
        if not self._workflows:
            return
        terminal = {
            wid: wf for wid, wf in self._workflows.items() if wf.status.is_terminal
        }
        pool = terminal or self._workflows
        oldest_id = min(pool, key=lambda k: pool[k].created_at)
        if not terminal:
            self._logger.warning(
                "Evicting active workflow '%s' to stay within max_workflows; "
                "no terminal-status workflow was available to evict instead.",
                oldest_id,
            )
        del self._workflows[oldest_id]


# ---------------------------------------------------------------------------
# Search scoring helpers
# ---------------------------------------------------------------------------


def _keyword_score(text: str, terms: list[str]) -> float:
    """
    Compute a simple relevance score for ``text`` against ``terms``.

    The score is the fraction of ``terms`` that appear as a
    case-insensitive substring of ``text``, in [0.0, 1.0].

    Args:
        text: The candidate text to score.
        terms: Pre-lowercased, non-empty query terms.

    Returns:
        A relevance score in [0.0, 1.0]. Returns 0.0 for empty text or
        an empty term list.
    """
    if not text or not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)


def _snippet(text: str, max_length: int = 160) -> str:
    """
    Produce a short excerpt of ``text`` for display in search results.

    Args:
        text: The source text.
        max_length: Maximum length of the returned snippet.

    Returns:
        ``text`` unchanged if within ``max_length``, otherwise a
        truncated copy ending in an ellipsis.
    """
    stripped = text.strip()
    if len(stripped) <= max_length:
        return stripped
    return stripped[: max_length - 1].rstrip() + "\u2026"


def _result_searchable_text(result: AgentResult) -> str:
    """
    Extract a plain-text representation of an ``AgentResult`` for
    keyword search.

    Args:
        result: The result to extract text from.

    Returns:
        A string combining the result's error message (if any) with a
        string form of its output, suitable for substring matching.
    """
    parts: list[str] = []
    if result.error is not None:
        parts.append(result.error.message)
    if isinstance(result.output, str):
        parts.append(result.output)
    elif result.output is not None:
        parts.append(str(result.output))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# MemoryManager facade
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Ergonomic facade over a ``MemoryStore``, intended to be held directly
    by ``BaseAgent`` subclasses and the orchestrator.

    Where ``MemoryStore`` exposes a generic CRUD-style contract,
    ``MemoryManager`` exposes the handful of higher-level operations an
    agent actually performs during a request: remembering a conversation
    turn, recording a completed task/result pair, and checkpointing
    workflow progress. It also owns the optional ``VectorMemoryBackend``
    used for semantic search, keeping that concern out of the storage
    contract itself.

    Attributes:
        store: The underlying ``MemoryStore`` implementation.
        vector_backend: Optional semantic-search backend. When ``None``,
            ``semantic_search`` raises ``AgentMemoryError`` with guidance
            rather than silently falling back to keyword search, so
            callers are not misled about the quality of the results.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        *,
        vector_backend: Optional[VectorMemoryBackend] = None,
    ) -> None:
        """
        Initialize the memory manager.

        Args:
            store: The backing ``MemoryStore``. Defaults to a new
                ``InMemoryMemoryStore`` with default ``MemoryConfig``.
            vector_backend: Optional embeddings-backed semantic search
                implementation.
        """
        self.store: MemoryStore = store or InMemoryMemoryStore()
        self.vector_backend: Optional[VectorMemoryBackend] = vector_backend
        self._logger = logging.getLogger(f"{__name__}.MemoryManager")

    # -- Conversation turns --------------------------------------------------

    def get_or_create_conversation(
        self, conversation_id: str, *, user_id: Optional[str] = None
    ) -> ConversationContext:
        """
        Fetch a conversation, creating it if it does not already exist.

        Args:
            conversation_id: Identifier of the conversation.
            user_id: Identifier of the principal driving the
                conversation, used only if the conversation is created.

        Returns:
            The existing or newly created ``ConversationContext``.
        """
        if self.store.conversation_exists(conversation_id):
            return self.store.get_conversation(conversation_id)
        return self.store.create_conversation(
            conversation_id=conversation_id, user_id=user_id
        )

    def remember_turn(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        agent_type: Optional[AgentType] = None,
        embedding_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentMemoryRecord:
        """
        Record a single conversation turn, creating the conversation if
        needed.

        If ``self.vector_backend`` is configured, the new record is
        indexed for semantic search and its ``embedding_id`` is set from
        the backend's result (unless one was explicitly supplied).

        Args:
            conversation_id: Identifier of the conversation this turn
                belongs to.
            role: Origin of the content, e.g. "user", "agent", "system",
                "tool".
            content: The text content of this turn.
            agent_type: The agent role that authored this turn, if any.
            embedding_id: Pre-computed embedding identifier for this
                content, if already known.
            metadata: Arbitrary additional attributes for this turn.

        Returns:
            The stored ``AgentMemoryRecord``.
        """
        self.get_or_create_conversation(conversation_id)
        record = AgentMemoryRecord(
            conversation_id=conversation_id,
            agent_type=agent_type,
            role=role,
            content=content,
            embedding_id=embedding_id,
            metadata=metadata or {},
        )
        if embedding_id is None and self.vector_backend is not None:
            try:
                generated_id = self.vector_backend.index_record(record)
                record = replace(record, embedding_id=generated_id)
            except Exception:  # noqa: BLE001 - indexing must not block memory writes
                self._logger.exception(
                    "Failed to index memory record '%s' with the configured "
                    "vector backend; continuing without an embedding_id.",
                    record.memory_id,
                )
        self.store.append_message(conversation_id, record)
        return record

    def recall(
        self, conversation_id: str, *, limit: int = 10
    ) -> list[AgentMemoryRecord]:
        """
        Retrieve the most recent turns in a conversation.

        Args:
            conversation_id: Identifier of the conversation.
            limit: Maximum number of turns to return.

        Returns:
            Up to ``limit`` recent ``AgentMemoryRecord`` entries, oldest
            to newest.
        """
        return self.store.get_recent_messages(conversation_id, limit=limit)

    # -- Tasks / results ------------------------------------------------

    def record_task(self, task: AgentTask) -> AgentTask:
        """
        Persist a task (typically once it starts or reaches a terminal
        status).

        Args:
            task: The task to persist.

        Returns:
            The stored ``AgentTask``.
        """
        return self.store.save_task(task)

    def record_result(self, result: AgentResult) -> AgentResult:
        """
        Persist an agent's output for a completed task.

        Args:
            result: The result to persist.

        Returns:
            The stored ``AgentResult``.
        """
        return self.store.save_result(result)

    def record_task_result(
        self, task: AgentTask, result: AgentResult
    ) -> tuple[AgentTask, AgentResult]:
        """
        Persist a task and its corresponding result together.

        Convenience wrapper around ``record_task`` and ``record_result``
        for the common case of an agent finishing ``execute`` for a
        single task.

        Args:
            task: The completed task.
            result: The result produced for that task.

        Returns:
            A ``(stored_task, stored_result)`` tuple.
        """
        stored_task = self.record_task(task)
        stored_result = self.record_result(result)
        return stored_task, stored_result

    # -- Workflows -------------------------------------------------------

    def checkpoint_workflow(self, workflow: WorkflowState) -> WorkflowState:
        """
        Persist the current state of a workflow.

        Intended to be called by the orchestrator after each meaningful
        state transition (a task starting, completing, or failing) so
        the workflow can be inspected or resumed.

        Args:
            workflow: The workflow state to persist.

        Returns:
            The stored ``WorkflowState``.
        """
        return self.store.save_workflow(workflow)

    def get_workflow(self, workflow_id: str) -> WorkflowState:
        """
        Retrieve a workflow's current state.

        Args:
            workflow_id: Identifier of the workflow.

        Returns:
            The stored ``WorkflowState``.

        Raises:
            MemoryNotFoundError: If no such workflow is stored.
        """
        return self.store.get_workflow(workflow_id)

    # -- Search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        record_types: Optional[Iterable[MemoryRecordType]] = None,
        conversation_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """
        Keyword-search stored memory records.

        Args:
            query: Free-text query.
            record_types: If provided, restrict the search to these
                record types.
            conversation_id: If provided, restrict conversation-message
                results to this conversation.
            limit: Maximum number of results to return.

        Returns:
            Matching ``MemorySearchResult`` instances, most relevant
            first.
        """
        return self.store.search(
            query,
            record_types=record_types,
            conversation_id=conversation_id,
            limit=limit,
        )

    def semantic_search(
        self, query: str, *, top_k: int = 5
    ) -> list[tuple[AgentMemoryRecord, float]]:
        """
        Semantically search conversation memory using the configured
        ``VectorMemoryBackend``.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(record, score)`` pairs ordered by descending
            similarity score.

        Raises:
            AgentMemoryError: If no ``VectorMemoryBackend`` has been
                configured for this manager. Use ``search`` for
                keyword-based recall instead, or construct this
                ``MemoryManager`` with a ``vector_backend`` (e.g. one
                built on ``embeddings.embedding_search``).
        """
        if self.vector_backend is None:
            raise AgentMemoryError(
                "semantic_search requires a VectorMemoryBackend, but none "
                "was configured on this MemoryManager. Use search() for "
                "keyword-based recall, or pass vector_backend= when "
                "constructing MemoryManager."
            )
        return self.vector_backend.semantic_search(query, top_k=top_k)

    # -- Maintenance -----------------------------------------------------

    def prune_expired(self) -> int:
        """Evict expired short-term records from the backing store."""
        return self.store.prune_expired()

    def stats(self) -> dict[str, int]:
        """Return record counts per collection from the backing store."""
        return self.store.stats()

    def clear(self) -> None:
        """Remove all records from the backing store."""
        self.store.clear()


__all__: list[str] = [
    "AgentMemoryError",
    "MemoryNotFoundError",
    "MemoryValidationError",
    "MemoryCapacityError",
    "MemoryRecordType",
    "MemoryScope",
    "MemoryConfig",
    "MemorySearchResult",
    "VectorMemoryBackend",
    "MemoryStore",
    "InMemoryMemoryStore",
    "MemoryManager",
    "to_serializable",
]