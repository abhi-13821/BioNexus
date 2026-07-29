"""
agents/conversation_manager.py

Conversation Manager for the BioNexus Multi-Agent AI System.

The ConversationManager is responsible for managing user conversations,
maintaining session state, tracking chat history, and interfacing with
the Coordinator for agent execution. It supports multiple simultaneous
conversations, follow-up questions, and conversation export.

The manager is designed to be:
    - Thread-safe: Supports concurrent conversations
    - Extensible: Easy to add new conversation features
    - Observable: Collects conversation metadata
    - Robust: Handles errors gracefully

Responsibilities:
    1. Maintain conversation sessions
    2. Manage chat history
    3. Track user context across multiple turns
    4. Interface with the Coordinator
    5. Support follow-up questions
    6. Maintain shared Memory
    7. Create and resume conversations
    8. Clear and export conversations
    9. Support multiple simultaneous sessions
    10. Return structured responses

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from agents.coordinator import Coordinator, CoordinatorResponse
from agents.memory import MemoryManager
from agents.models import AgentType, TaskStatus

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------


class ConversationStatus(str, Enum):
    """Status of a conversation session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


class ConversationRole(str, Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ----------------------------------------------------------------------
# Conversation models
# ----------------------------------------------------------------------


@dataclass
class ConversationMessage:
    """
    A single message in a conversation.

    Attributes:
        message_id: Unique identifier for this message.
        conversation_id: Identifier of the conversation.
        role: The role of the message sender.
        content: The message content.
        timestamp: When the message was created.
        metadata: Additional metadata for the message.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    role: ConversationRole = ConversationRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create a ConversationMessage from a dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            conversation_id=data.get("conversation_id", ""),
            role=ConversationRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSession:
    """
    A conversation session.

    Attributes:
        conversation_id: Unique identifier for the conversation.
        user_id: Identifier of the user.
        title: Human-readable title for the conversation.
        status: Current status of the conversation.
        messages: List of messages in the conversation.
        context: Additional context for the conversation.
        created_at: When the conversation was created.
        updated_at: When the conversation was last updated.
        completed_at: When the conversation was completed.
        metadata: Additional metadata for the conversation.
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    title: str = "New Conversation"
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list[ConversationMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the conversation."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string.")

    def add_message(self, message: ConversationMessage) -> None:
        """
        Add a message to the conversation.

        Args:
            message: The message to add.
        """
        message.conversation_id = self.conversation_id
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_last_message(self) -> Optional[ConversationMessage]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def get_messages_by_role(self, role: ConversationRole) -> list[ConversationMessage]:
        """Get all messages with a specific role."""
        return [m for m in self.messages if m.role == role]

    def get_user_messages(self) -> list[ConversationMessage]:
        """Get all user messages."""
        return self.get_messages_by_role(ConversationRole.USER)

    def get_assistant_messages(self) -> list[ConversationMessage]:
        """Get all assistant messages."""
        return self.get_messages_by_role(ConversationRole.ASSISTANT)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status.value,
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSession":
        """Create a ConversationSession from a dictionary."""
        return cls(
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            title=data.get("title", "New Conversation"),
            status=ConversationStatus(data.get("status", "active")),
            messages=[ConversationMessage.from_dict(m) for m in data.get("messages", [])],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationResponse:
    """
    Response from the ConversationManager.

    Attributes:
        response_id: Unique identifier for this response.
        conversation_id: Identifier of the conversation.
        user_message: The user's message that was processed.
        assistant_message: The assistant's response message.
        coordinator_response: The raw CoordinatorResponse.
        status: Overall status of the response.
        message: Human-readable summary message.
        error: Error information if the response failed.
        created_at: When the response was created.
    """

    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    user_message: Optional[ConversationMessage] = None
    assistant_message: Optional[ConversationMessage] = None
    coordinator_response: Optional[CoordinatorResponse] = None
    status: TaskStatus = TaskStatus.COMPLETED
    message: str = ""
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "response_id": self.response_id,
            "conversation_id": self.conversation_id,
            "user_message": self.user_message.to_dict() if self.user_message else None,
            "assistant_message": self.assistant_message.to_dict() if self.assistant_message else None,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }


# ----------------------------------------------------------------------
# ConversationManager configuration
# ----------------------------------------------------------------------


@dataclass
class ConversationManagerConfig:
    """
    Configuration for the ConversationManager.

    Attributes:
        max_conversations_per_user: Maximum conversations per user.
        max_messages_per_conversation: Maximum messages per conversation.
        conversation_timeout_seconds: Timeout for inactive conversations.
        enable_persistence: Whether to persist conversations.
        persistence_path: Path for persistence storage.
        auto_save_interval_seconds: Interval for auto-saving conversations.
        default_title: Default title for new conversations.
    """

    max_conversations_per_user: int = 100
    max_messages_per_conversation: int = 500
    conversation_timeout_seconds: float = 3600.0  # 1 hour
    enable_persistence: bool = False
    persistence_path: str = "data/conversations.json"
    auto_save_interval_seconds: float = 60.0
    default_title: str = "New Conversation"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_conversations_per_user < 1:
            raise ValueError("max_conversations_per_user must be positive.")
        if self.max_messages_per_conversation < 1:
            raise ValueError("max_messages_per_conversation must be positive.")
        if self.conversation_timeout_seconds < 0:
            raise ValueError("conversation_timeout_seconds cannot be negative.")


# ----------------------------------------------------------------------
# ConversationManager
# ----------------------------------------------------------------------


class ConversationManager:
    """
    Manages user conversations for the BioNexus Multi-Agent AI system.

    The ConversationManager maintains conversation sessions, chat history,
    user context, and interfaces with the Coordinator for agent execution.
    It supports multiple simultaneous conversations, follow-up questions,
    and conversation export.

    Example:
        >>> manager = ConversationManager(coordinator)
        >>> await manager.initialize()
        >>> response = await manager.process_message(
        ...     conversation_id="conv_123",
        ...     user_message="Find EGFR papers",
        ... )
        >>> print(response.message)

    Attributes:
        coordinator: The Coordinator instance.
        memory: Shared MemoryManager instance.
        config: ConversationManager configuration.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        memory: Optional[MemoryManager] = None,
        config: Optional[ConversationManagerConfig] = None,
    ) -> None:
        """
        Initialize the ConversationManager.

        Args:
            coordinator: The Coordinator instance.
            memory: Shared MemoryManager. If not provided, one is created.
            config: ConversationManager configuration.

        Raises:
            ValueError: If coordinator is not provided.
        """
        if coordinator is None:
            raise ValueError("coordinator is required.")

        self.coordinator: Coordinator = coordinator
        self.memory: MemoryManager = memory or MemoryManager()
        self.config: ConversationManagerConfig = config or ConversationManagerConfig()

        # In-memory session storage
        self._sessions: dict[str, ConversationSession] = {}
        self._user_sessions: dict[str, list[str]] = {}  # user_id -> list of conversation_ids

        self._initialized: bool = False
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.ConversationManager")

    async def initialize(self) -> None:
        """
        Initialize the ConversationManager.

        This loads persisted conversations if enabled.

        Raises:
            Exception: If initialization fails.
        """
        if self._initialized:
            return

        try:
            # Load persisted conversations
            if self.config.enable_persistence:
                await self._load_conversations()

            self._initialized = True
            self._logger.info("ConversationManager initialized successfully.")

        except Exception as e:
            self._logger.error("ConversationManager initialization failed: %s", e)
            raise

    async def shutdown(self) -> None:
        """Shut down the ConversationManager and save conversations."""
        self._logger.info("Shutting down ConversationManager...")

        # Save conversations
        if self.config.enable_persistence:
            await self._save_conversations()

        self._initialized = False
        self._logger.info("ConversationManager shut down successfully.")

    async def create_conversation(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ConversationSession:
        """
        Create a new conversation.

        Args:
            user_id: Optional user identifier.
            title: Optional conversation title.
            context: Optional initial context.

        Returns:
            The new ConversationSession.

        Raises:
            ValueError: If user has too many active conversations.
        """
        async with self._lock:
            # Check if user has too many conversations
            if user_id:
                user_conversations = self._user_sessions.get(user_id, [])
                active_count = sum(
                    1 for cid in user_conversations
                    if cid in self._sessions and self._sessions[cid].status == ConversationStatus.ACTIVE
                )
                if active_count >= self.config.max_conversations_per_user:
                    raise ValueError(
                        f"User has reached the maximum of {self.config.max_conversations_per_user} "
                        "active conversations. Please complete or clear an existing conversation."
                    )

            # Create the session
            session = ConversationSession(
                user_id=user_id,
                title=title or self.config.default_title,
                context=context or {},
            )

            # Store the session
            self._sessions[session.conversation_id] = session

            # Add to user sessions
            if user_id:
                if user_id not in self._user_sessions:
                    self._user_sessions[user_id] = []
                self._user_sessions[user_id].append(session.conversation_id)

            self._logger.info(
                "Created conversation %s for user %s",
                session.conversation_id[:8],
                user_id or "anonymous",
            )

            return session

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The ConversationSession or None if not found.
        """
        async with self._lock:
            session = self._sessions.get(conversation_id)

            # Check if conversation has expired
            if session and session.status == ConversationStatus.ACTIVE:
                if self._is_expired(session):
                    session.status = ConversationStatus.EXPIRED
                    self._logger.info("Conversation %s expired", conversation_id[:8])

            return session

    async def get_conversations_by_user(self, user_id: str) -> list[ConversationSession]:
        """
        Get all conversations for a user.

        Args:
            user_id: The user ID.

        Returns:
            A list of ConversationSession instances.
        """
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            return [self._sessions[cid] for cid in session_ids if cid in self._sessions]

    async def process_message(
        self,
        conversation_id: str,
        user_message: str,
        user_id: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> ConversationResponse:
        """
        Process a user message in a conversation.

        Args:
            conversation_id: The conversation ID.
            user_message: The user's message.
            user_id: Optional user ID for authentication.
            parameters: Optional additional parameters.

        Returns:
            A ConversationResponse.

        Raises:
            ValueError: If the conversation is not found or invalid.
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager is not initialized. Call initialize() first.")

        # Get or create the conversation
        session = await self.get_conversation(conversation_id)

        if session is None:
            # Create a new conversation
            session = await self.create_conversation(
                user_id=user_id,
                title=user_message[:50] + ("..." if len(user_message) > 50 else ""),
            )

        # Check if conversation is active
        if session.status != ConversationStatus.ACTIVE:
            if session.status == ConversationStatus.COMPLETED:
                raise ValueError("Conversation has been completed.")
            elif session.status == ConversationStatus.EXPIRED:
                raise ValueError("Conversation has expired. Please start a new conversation.")
            else:
                raise ValueError(f"Conversation is {session.status.value}.")

        # Check message limit
        if len(session.messages) >= self.config.max_messages_per_conversation:
            raise ValueError(
                f"Conversation has reached the maximum of {self.config.max_messages_per_conversation} messages."
            )

        # Create user message
        user_msg = ConversationMessage(
            conversation_id=conversation_id,
            role=ConversationRole.USER,
            content=user_message,
            metadata={"user_id": user_id} if user_id else {},
        )

        # Add to session
        session.add_message(user_msg)

        # Update context
        session.context["last_user_message"] = user_message
        session.context["last_user_message_time"] = datetime.now(timezone.utc).isoformat()

        self._logger.info(
            "Processing message in conversation %s: '%s'",
            conversation_id[:8],
            user_message[:50],
        )

        try:
            # Process with coordinator
            coordinator_response = await self.coordinator.process_query(
                query=user_message,
                conversation_id=conversation_id,
                parameters=parameters or {},
                requested_by=user_id,
            )

            # Create assistant message
            assistant_content = self._format_coordinator_response(coordinator_response)
            assistant_msg = ConversationMessage(
                conversation_id=conversation_id,
                role=ConversationRole.ASSISTANT,
                content=assistant_content,
                metadata={
                    "status": coordinator_response.status.value,
                    "agent_count": len(coordinator_response.agent_results) if coordinator_response.agent_results else 0,
                    "confidence": coordinator_response.confidence.value if coordinator_response.confidence else None,
                },
            )

            # Add to session
            session.add_message(assistant_msg)

            # Update context
            session.context["last_response_status"] = coordinator_response.status.value
            session.context["last_assistant_message_time"] = datetime.now(timezone.utc).isoformat()

            # If the coordinator indicates completion, mark conversation as completed
            if coordinator_response.status == TaskStatus.COMPLETED:
                # Check if all agents succeeded
                all_success = all(
                    r.success for r in coordinator_response.agent_results
                ) if coordinator_response.agent_results else True

                if all_success:
                    session.status = ConversationStatus.COMPLETED
                    session.completed_at = datetime.now(timezone.utc)

            # Save conversations if persistence is enabled
            if self.config.enable_persistence:
                await self._save_conversations()

            return ConversationResponse(
                conversation_id=conversation_id,
                user_message=user_msg,
                assistant_message=assistant_msg,
                coordinator_response=coordinator_response,
                status=coordinator_response.status,
                message=coordinator_response.message,
                created_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            self._logger.error("Error processing message in conversation %s: %s", conversation_id[:8], e)

            # Create error response
            error_msg = ConversationMessage(
                conversation_id=conversation_id,
                role=ConversationRole.ASSISTANT,
                content=f"I encountered an error while processing your request: {str(e)}",
                metadata={"error": str(e)},
            )
            session.add_message(error_msg)

            return ConversationResponse(
                conversation_id=conversation_id,
                user_message=user_msg,
                assistant_message=error_msg,
                coordinator_response=None,
                status=TaskStatus.FAILED,
                message=f"Processing failed: {str(e)}",
                error=str(e),
                created_at=datetime.now(timezone.utc),
            )

    def _format_coordinator_response(self, response: CoordinatorResponse) -> str:
        """
        Format a CoordinatorResponse for display to the user.

        Args:
            response: The CoordinatorResponse.

        Returns:
            A formatted string.
        """
        if not response:
            return "I couldn't generate a response. Please try again."

        if response.status == TaskStatus.COMPLETED:
            return response.message

        elif response.status == TaskStatus.FAILED:
            errors = []
            if response.errors:
                for error in response.errors:
                    errors.append(f"- {error.message}")
            if errors:
                return f"Some issues occurred:\n\n{chr(10).join(errors)}"
            return response.message or "I encountered an issue. Please try again."

        elif response.status == TaskStatus.WAITING_ON_DEPENDENCY:
            return response.message or "I need more information. Please clarify your request."

        else:
            return response.message or "I've processed your request."

    def _is_expired(self, session: ConversationSession) -> bool:
        """
        Check if a conversation has expired.

        Args:
            session: The conversation session.

        Returns:
            True if the conversation has expired.
        """
        if self.config.conversation_timeout_seconds <= 0:
            return False

        delta = datetime.now(timezone.utc) - session.updated_at
        return delta.total_seconds() > self.config.conversation_timeout_seconds

    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation (remove all messages).

        Args:
            conversation_id: The conversation ID.

        Returns:
            True if the conversation was cleared, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(conversation_id)

            if session is None:
                return False

            session.messages = []
            session.updated_at = datetime.now(timezone.utc)

            self._logger.info("Cleared conversation %s", conversation_id[:8])

            if self.config.enable_persistence:
                await self._save_conversations()

            return True

    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation permanently.

        Args:
            conversation_id: The conversation ID.

        Returns:
            True if the conversation was deleted, False otherwise.
        """
        async with self._lock:
            session = self._sessions.pop(conversation_id, None)

            if session is None:
                return False

            # Remove from user sessions
            if session.user_id and session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    cid for cid in self._user_sessions[session.user_id]
                    if cid != conversation_id
                ]

            self._logger.info("Deleted conversation %s", conversation_id[:8])

            if self.config.enable_persistence:
                await self._save_conversations()

            return True

    async def export_conversation(self, conversation_id: str, format: str = "json") -> Optional[str]:
        """
        Export a conversation in the specified format.

        Args:
            conversation_id: The conversation ID.
            format: The export format ('json' or 'text').

        Returns:
            The exported conversation as a string, or None if not found.
        """
        session = await self.get_conversation(conversation_id)

        if session is None:
            return None

        if format == "json":
            return json.dumps(session.to_dict(), indent=2)
        elif format == "text":
            lines = [
                f"Conversation: {session.title}",
                f"ID: {session.conversation_id}",
                f"User: {session.user_id or 'Anonymous'}",
                f"Created: {session.created_at.isoformat()}",
                f"Updated: {session.updated_at.isoformat()}",
                f"Status: {session.status.value}",
                "",
                "--- Messages ---",
                "",
            ]
            for msg in session.messages:
                role = msg.role.value.upper()
                lines.append(f"[{role}] {msg.content}")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def get_conversation_summary(self, conversation_id: str) -> Optional[dict[str, Any]]:
        """
        Get a summary of a conversation.

        Args:
            conversation_id: The conversation ID.

        Returns:
            A dictionary with summary information, or None if not found.
        """
        session = await self.get_conversation(conversation_id)

        if session is None:
            return None

        return {
            "conversation_id": session.conversation_id,
            "title": session.title,
            "status": session.status.value,
            "message_count": len(session.messages),
            "user_message_count": len(session.get_user_messages()),
            "assistant_message_count": len(session.get_assistant_messages()),
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        }

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSession]:
        """
        List conversations with optional filtering.

        Args:
            user_id: Optional user ID filter.
            status: Optional status filter.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            A list of ConversationSession instances.
        """
        async with self._lock:
            sessions = list(self._sessions.values())

            # Filter by user
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]

            # Filter by status
            if status:
                sessions = [s for s in sessions if s.status == status]

            # Sort by updated_at (newest first)
            sessions.sort(key=lambda s: s.updated_at, reverse=True)

            # Apply pagination
            return sessions[offset:offset + limit]

    async def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Update the title of a conversation.

        Args:
            conversation_id: The conversation ID.
            title: The new title.

        Returns:
            True if the title was updated, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(conversation_id)

            if session is None:
                return False

            session.title = title
            session.updated_at = datetime.now(timezone.utc)

            self._logger.info("Updated title for conversation %s to '%s'", conversation_id[:8], title)

            if self.config.enable_persistence:
                await self._save_conversations()

            return True

    async def add_context(self, conversation_id: str, key: str, value: Any) -> bool:
        """
        Add context to a conversation.

        Args:
            conversation_id: The conversation ID.
            key: The context key.
            value: The context value.

        Returns:
            True if the context was added, False otherwise.
        """
        async with self._lock:
            session = self._sessions.get(conversation_id)

            if session is None:
                return False

            session.context[key] = value
            session.updated_at = datetime.now(timezone.utc)

            return True

    async def get_context(self, conversation_id: str, key: str) -> Any:
        """
        Get context from a conversation.

        Args:
            conversation_id: The conversation ID.
            key: The context key.

        Returns:
            The context value, or None if not found.
        """
        session = await self.get_conversation(conversation_id)

        if session is None:
            return None

        return session.context.get(key)

    async def _save_conversations(self) -> None:
        """Save conversations to disk."""
        if not self.config.enable_persistence:
            return

        try:
            import os

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.persistence_path), exist_ok=True)

            data = {
                "sessions": {cid: session.to_dict() for cid, session in self._sessions.items()},
                "user_sessions": self._user_sessions,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.config.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            self._logger.debug("Saved %d conversations to %s", len(self._sessions), self.config.persistence_path)

        except Exception as e:
            self._logger.error("Failed to save conversations: %s", e)

    async def _load_conversations(self) -> None:
        """Load conversations from disk."""
        if not self.config.enable_persistence:
            return

        try:
            import os

            if not os.path.exists(self.config.persistence_path):
                self._logger.info("No persisted conversations found at %s", self.config.persistence_path)
                return

            with open(self.config.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sessions_data = data.get("sessions", {})
            for cid, session_data in sessions_data.items():
                session = ConversationSession.from_dict(session_data)
                self._sessions[cid] = session

            self._user_sessions = data.get("user_sessions", {})

            self._logger.info("Loaded %d conversations from %s", len(self._sessions), self.config.persistence_path)

        except Exception as e:
            self._logger.error("Failed to load conversations: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the conversation manager.

        Returns:
            A dictionary with statistics.
        """
        total_messages = sum(len(s.messages) for s in self._sessions.values())
        active_sessions = sum(1 for s in self._sessions.values() if s.status == ConversationStatus.ACTIVE)
        completed_sessions = sum(1 for s in self._sessions.values() if s.status == ConversationStatus.COMPLETED)
        expired_sessions = sum(1 for s in self._sessions.values() if s.status == ConversationStatus.EXPIRED)

        return {
            "total_conversations": len(self._sessions),
            "active_conversations": active_sessions,
            "completed_conversations": completed_sessions,
            "expired_conversations": expired_sessions,
            "total_messages": total_messages,
            "average_messages_per_conversation": total_messages / len(self._sessions) if self._sessions else 0,
            "total_users": len(self._user_sessions),
            "initialized": self._initialized,
        }

    async def __aenter__(self) -> "ConversationManager":
        """Support async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Support async context manager."""
        await self.shutdown()


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------


def create_conversation_manager(
    coordinator: Coordinator,
    memory: Optional[MemoryManager] = None,
    config: Optional[ConversationManagerConfig] = None,
) -> ConversationManager:
    """
    Factory function to create a ConversationManager instance.

    Args:
        coordinator: The Coordinator instance.
        memory: Shared MemoryManager.
        config: ConversationManager configuration.

    Returns:
        A configured ConversationManager instance.
    """
    return ConversationManager(
        coordinator=coordinator,
        memory=memory,
        config=config,
    )


__all__: list[str] = [
    "ConversationStatus",
    "ConversationRole",
    "ConversationMessage",
    "ConversationSession",
    "ConversationResponse",
    "ConversationManagerConfig",
    "ConversationManager",
    "create_conversation_manager",
]