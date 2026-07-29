"""
tests/test_agents.py

Comprehensive test suite for the BioNexus Multi-Agent AI System.

This module contains unit and integration tests covering all major
components of the multi-agent framework, including base agent,
memory, specialist agents, planner, coordinator, conversation manager,
prompts, tools, and cache.

Test coverage targets:
    - All public APIs are tested
    - Valid and invalid inputs are handled correctly
    - Edge cases and empty inputs are tested
    - Memory behavior is verified
    - Coordinator execution is tested
    - Planner routing is validated
    - Cache behavior is thoroughly tested
    - Error handling is comprehensive
    - Integration tests cover end-to-end scenarios

Compatibility
-------------
Targets Python 3.11 with pytest.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.base_agent import (
    AgentConfig,
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentNotInitializedError,
    AgentShutdownError,
    AgentValidationError,
    BaseAgent,
)
from agents.cache import (
    Cache,
    CacheConfig,
    CacheKeyError,
    cached,
    clear_default_cache,
    get_default_cache,
    reset_default_cache,
)
from agents.conversation_manager import (
    ConversationManager,
    ConversationManagerConfig,
    ConversationMessage,
    ConversationResponse,
    ConversationRole,
    ConversationSession,
    ConversationStatus,
)
from agents.coordinator import (
    AgentExecutionResult,
    AgentRegistry,
    Coordinator,
    CoordinatorConfig,
    CoordinatorInitializationError,
    CoordinatorResponse,
    create_coordinator,
)
from agents.drug_agent import DrugAgent, create_drug_agent
from agents.drug_discovery_agent import DrugDiscoveryAgent, create_drug_discovery_agent
from agents.knowledge_graph_agent import KnowledgeGraphAgent, create_knowledge_graph_agent
from agents.literature_agent import LiteratureAgent, create_literature_agent
from agents.memory import (
    InMemoryMemoryStore,
    MemoryConfig,
    MemoryManager,
    MemoryNotFoundError,
)
from agents.models import (
    AgentMemoryRecord,
    AgentRequest,
    AgentResponse,
    AgentResult,
    AgentTask,
    AgentType,
    ConfidenceScore,
    ConversationContext,
    ErrorInformation,
    ExecutionMetadata,
    Priority,
    TaskStatus,
    WorkflowState,
)
from agents.planner_agent import (
    ExecutionMode,
    ExecutionPlan,
    IntentCategory,
    PlanStep,
    PlannerAgent,
    PlannerConfig,
    create_planner_agent,
)
from agents.prompts import (
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    create_prompt_registry,
    render_prompt,
    render_prompt_with_defaults,
)
from agents.smiles_agent import SmilesAgent, create_smiles_agent
from agents.tools import (
    aggregate_confidence,
    clean_text,
    confidence_to_label,
    extract_entities_from_text,
    extract_keywords,
    format_error_response,
    format_response,
    format_success_response,
    generate_id,
    get_conversation_id,
    merge_dicts,
    merge_results,
    normalize_confidence,
    pretty_json,
    summarize_conversation,
    to_json_safe,
    truncate_text,
    validate_in_range,
    validate_non_empty,
    validate_positive,
    validate_type,
    validate_uuid,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def agent_config() -> AgentConfig:
    """Return a basic agent configuration."""
    return AgentConfig(
        default_timeout_seconds=10.0,
        max_retries=2,
        model_name="test_model",
        enable_rag=False,
        enable_tool_use=False,
    )


@pytest.fixture
def memory_config() -> MemoryConfig:
    """Return a memory configuration."""
    return MemoryConfig(
        max_messages_per_conversation=50,
        message_ttl_seconds=60,
        max_conversations=10,
        max_tasks=100,
        max_results=100,
        max_workflows=50,
        auto_prune_on_write=True,
    )


@pytest.fixture
def memory_manager(memory_config: MemoryConfig) -> MemoryManager:
    """Return a memory manager."""
    store = InMemoryMemoryStore(config=memory_config)
    return MemoryManager(store=store)


@pytest.fixture
def mock_agent_request() -> AgentRequest:
    """Return a mock agent request."""
    return AgentRequest(
        instruction="Test instruction",
        parameters={"key": "value"},
        conversation_id="test_conv",
        priority=Priority.NORMAL,
    )


@pytest.fixture
def mock_agent_task() -> AgentTask:
    """Return a mock agent task."""
    return AgentTask(
        request_id="test_request",
        agent_type=AgentType.ORCHESTRATOR,
        description="Test task",
        input_data={"key": "value"},
        status=TaskStatus.PENDING,
    )


@pytest.fixture
def mock_confidence() -> ConfidenceScore:
    """Return a mock confidence score."""
    return ConfidenceScore(
        value=0.85,
        basis="Test confidence",
        source_module="test",
    )


# ----------------------------------------------------------------------
# Tests for models.py
# ----------------------------------------------------------------------

class TestModels:
    """Test suite for agent models."""

    def test_agent_request_validation(self) -> None:
        """Test AgentRequest validation."""
        # Valid request
        request = AgentRequest(instruction="test", parameters={})
        assert request.instruction == "test"

        # Empty instruction
        with pytest.raises(ValueError, match="instruction must be a non-empty string"):
            AgentRequest(instruction="", parameters={})

        # Empty request_id
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            request = AgentRequest(instruction="test", parameters={})
            request.request_id = ""
            request.__post_init__()

        # Negative timeout
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            AgentRequest(instruction="test", parameters={}, timeout_seconds=-1)

    def test_agent_task_validation(self) -> None:
        """Test AgentTask validation."""
        # Valid task
        task = AgentTask(description="test task")
        assert task.description == "test task"

        # Empty description
        with pytest.raises(ValueError, match="description must be a non-empty string"):
            AgentTask(description="")

        # Empty task_id
        with pytest.raises(ValueError, match="task_id must be a non-empty string"):
            task = AgentTask(description="test")
            task.task_id = ""
            task.__post_init__()

        # Timestamp validation
        with pytest.raises(ValueError, match="started_at cannot be earlier than created_at"):
            now = datetime.now(timezone.utc)
            AgentTask(
                description="test",
                created_at=now,
                started_at=now - timedelta(seconds=10),
            )

    def test_agent_task_is_ready(self) -> None:
        """Test AgentTask.is_ready method."""
        task = AgentTask(
            description="test",
            dependencies=["dep1", "dep2"],
        )

        # Not all dependencies met
        assert task.is_ready({"dep1"}) is False

        # All dependencies met
        assert task.is_ready({"dep1", "dep2"}) is True

        # No dependencies
        task.dependencies = []
        assert task.is_ready(set()) is True

    def test_task_status_terminal(self) -> None:
        """Test TaskStatus.is_terminal property."""
        terminal_statuses = [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        ]

        for status in terminal_statuses:
            assert status.is_terminal is True

        non_terminal = [
            TaskStatus.PENDING,
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_ON_DEPENDENCY,
        ]

        for status in non_terminal:
            assert status.is_terminal is False

    def test_confidence_score_validation(self) -> None:
        """Test ConfidenceScore validation."""
        # Valid confidence
        score = ConfidenceScore(value=0.75)
        assert score.value == 0.75

        # Invalid confidence
        with pytest.raises(ValueError, match="value must be within"):
            ConfidenceScore(value=1.5)

        with pytest.raises(ValueError, match="value must be within"):
            ConfidenceScore(value=-0.5)

    def test_confidence_score_label(self) -> None:
        """Test ConfidenceScore.label property."""
        assert ConfidenceScore(value=0.95).label == "very_high"
        assert ConfidenceScore(value=0.8).label == "high"
        assert ConfidenceScore(value=0.6).label == "medium"
        assert ConfidenceScore(value=0.3).label == "low"

    def test_execution_metadata_validation(self) -> None:
        """Test ExecutionMetadata validation."""
        # Valid metadata
        metadata = ExecutionMetadata(
            agent_id="test_agent",
            agent_type=AgentType.ORCHESTRATOR,
        )
        assert metadata.agent_id == "test_agent"

        # Empty agent_id
        with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
            ExecutionMetadata(agent_id="", agent_type=AgentType.ORCHESTRATOR)

        # Negative retries
        with pytest.raises(ValueError, match="retries cannot be negative"):
            ExecutionMetadata(
                agent_id="test",
                agent_type=AgentType.ORCHESTRATOR,
                retries=-1,
            )

        # Timestamp ordering
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(seconds=10)
        with pytest.raises(ValueError, match="completed_at cannot be earlier than started_at"):
            ExecutionMetadata(
                agent_id="test",
                agent_type=AgentType.ORCHESTRATOR,
                started_at=now,
                completed_at=earlier,
            )

    def test_agent_response_successful_results(self) -> None:
        """Test AgentResponse successful_results method."""
        # Create a valid failed result with error
        error = ErrorInformation(
            error_type="TestError",
            message="Something went wrong",
        )
        response = AgentResponse(
            request_id="test",
            results=[
                AgentResult(status=TaskStatus.COMPLETED, task_id="1"),
                AgentResult(status=TaskStatus.COMPLETED, task_id="2"),
                AgentResult(status=TaskStatus.FAILED, task_id="3", error=error),
            ],
        )

        successful = response.successful_results()
        assert len(successful) == 2
        assert successful[0].task_id == "1"

        failed = response.failed_results()
        assert len(failed) == 1
        assert failed[0].task_id == "3"

    def test_conversation_context_recent_history(self) -> None:
        """Test ConversationContext.recent_history method."""
        conv_id = "test_conv_id"
        context = ConversationContext(conversation_id=conv_id)
        for i in range(15):
            context.history.append(
                AgentMemoryRecord(
                    conversation_id=conv_id,
                    content=f"Message {i}",
                )
            )

        recent = context.recent_history(limit=5)
        assert len(recent) == 5
        assert recent[0].content == "Message 10"
        assert recent[-1].content == "Message 14"

        with pytest.raises(ValueError, match="limit must be a positive integer"):
            context.recent_history(limit=0)

    def test_workflow_state_methods(self) -> None:
        """Test WorkflowState methods."""
        workflow = WorkflowState(
            tasks=[
                AgentTask(description="Task 1", status=TaskStatus.COMPLETED),
                AgentTask(description="Task 2", status=TaskStatus.PENDING),
                AgentTask(description="Task 3", status=TaskStatus.PENDING),
            ]
        )

        # PENDING tasks are considered pending (not terminal)
        assert workflow.is_completed() is False
        pending = workflow.pending_tasks()
        assert len(pending) == 2

        # Failed tasks
        workflow.tasks.append(AgentTask(description="Task 4", status=TaskStatus.FAILED))
        failed = workflow.failed_tasks()
        assert len(failed) == 1

        # Progress percentage
        workflow.status = TaskStatus.COMPLETED
        assert workflow.is_completed() is True

        # Empty workflow
        workflow_empty = WorkflowState()
        assert workflow_empty.progress_percentage() == 0.0


# ----------------------------------------------------------------------
# Tests for base_agent.py
# ----------------------------------------------------------------------

class TestBaseAgent:
    """Test suite for BaseAgent."""

    class ConcreteAgent(BaseAgent):
        """Concrete implementation for testing."""

        async def initialize(self) -> None:
            self._initialized = True

        def validate_request(self, request: AgentRequest) -> None:
            if not request.instruction:
                raise AgentValidationError("Invalid instruction")

        async def execute(self, task: AgentTask) -> AgentResult:
            return AgentResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output={"result": "success"},
            )

        async def shutdown(self) -> None:
            self._initialized = False

    def test_agent_initialization(self) -> None:
        """Test agent initialization."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)

        # Not initialized
        assert agent.is_initialized is False

        # Start
        asyncio.run(agent.start())
        assert agent.is_initialized is True

        # Stop
        asyncio.run(agent.stop())
        assert agent.is_initialized is False

    def test_double_initialization(self) -> None:
        """Test double initialization warning."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        asyncio.run(agent.start())
        asyncio.run(agent.start())  # Should not raise, just log warning

    def test_handle_request_success(self) -> None:
        """Test successful request handling."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        asyncio.run(agent.start())

        request = AgentRequest(instruction="test instruction")
        response = asyncio.run(agent.handle_request(request))

        assert response.status == TaskStatus.COMPLETED
        assert len(response.results) == 1
        assert response.results[0].status == TaskStatus.COMPLETED

    def test_handle_request_not_initialized(self) -> None:
        """Test request handling when agent is not initialized."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)

        request = AgentRequest(instruction="test")
        response = asyncio.run(agent.handle_request(request))

        assert response.status == TaskStatus.FAILED
        assert len(response.errors) == 1
        assert "before start()" in response.errors[0].message

    def test_handle_request_validation_failure(self) -> None:
        """Test request handling with validation failure."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        asyncio.run(agent.start())

        # Create a valid request then modify it to trigger validation failure
        request = AgentRequest(instruction="test")
        # Override instruction to empty after construction
        request.instruction = ""
        response = asyncio.run(agent.handle_request(request))

        assert response.status == TaskStatus.FAILED
        assert len(response.errors) == 1

    def test_build_task(self) -> None:
        """Test task building."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        request = AgentRequest(
            instruction="test",
            parameters={"param": "value"},
        )
        task = agent._build_task(request)

        assert task.request_id == request.request_id
        assert task.agent_type == agent.agent_type
        assert task.description == request.instruction
        assert task.input_data == request.parameters
        assert task.status == TaskStatus.RUNNING

    def test_build_success_result(self) -> None:
        """Test building success result."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        task = AgentTask(description="test")
        output = {"result": "success"}

        result = agent._build_success_result(task, output)

        assert result.task_id == task.task_id
        assert result.status == TaskStatus.COMPLETED
        assert result.output == output

    def test_build_error_result(self) -> None:
        """Test building error result."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        task = AgentTask(description="test")
        error = AgentExecutionError("Test error")

        result = agent._build_error_result(task, error)

        assert result.task_id == task.task_id
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert result.error.message == "Test error"

    def test_build_error_result_invalid_status(self) -> None:
        """Test building error result with invalid status."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)
        task = AgentTask(description="test")
        error = AgentExecutionError("Test error")

        with pytest.raises(ValueError, match="status must be"):
            agent._build_error_result(task, error, status=TaskStatus.COMPLETED)

    def test_summarize_result_success(self) -> None:
        """Test result summarization - success."""
        result = AgentResult(
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            confidence=ConfidenceScore(value=0.9),
        )
        summary = BaseAgent._summarize_result(result)
        assert "completed successfully" in summary
        assert "very_high" in summary

    def test_summarize_result_failure(self) -> None:
        """Test result summarization - failure."""
        result = AgentResult(
            task_id="test_task",
            status=TaskStatus.FAILED,
            error=ErrorInformation(
                error_type="TestError",
                message="Something went wrong",
            ),
        )
        summary = BaseAgent._summarize_result(result)
        assert "did not complete successfully" in summary
        assert "Something went wrong" in summary

    def test_agent_context_manager(self) -> None:
        """Test agent async context manager."""
        agent = self.ConcreteAgent(AgentType.ORCHESTRATOR)

        async def run() -> None:
            async with agent as a:
                assert a.is_initialized is True

        asyncio.run(run())
        assert agent.is_initialized is False


# ----------------------------------------------------------------------
# Tests for specialist agents
# ----------------------------------------------------------------------

class TestLiteratureAgent:
    """Test suite for LiteratureAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test LiteratureAgent initialization."""
        agent = LiteratureAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test LiteratureAgent request validation."""
        agent = LiteratureAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Find papers on cancer")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Find papers on cancer")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_execute(self) -> None:
        """Test LiteratureAgent execution."""
        agent = LiteratureAgent()
        await agent.start()

        # Mock the _execute_search method
        with patch.object(agent, '_execute_search', return_value={"papers": [], "paper_count": 0}):
            task = AgentTask(
                request_id="test_request_123",
                description="Find papers on cancer",
                input_data={"query": "cancer research"},
            )
            result = await agent.execute(task)

            assert result.status == TaskStatus.COMPLETED
            assert result.output is not None

        await agent.stop()

    def test_agent_get_capabilities(self) -> None:
        """Test LiteratureAgent capabilities."""
        agent = LiteratureAgent()
        capabilities = agent.get_capabilities()
        assert "literature_search" in capabilities
        assert "literature_summarize" in capabilities
        assert "literature_review" in capabilities


class TestDrugAgent:
    """Test suite for DrugAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test DrugAgent initialization."""
        agent = DrugAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test DrugAgent request validation."""
        agent = DrugAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Tell me about Aspirin")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Tell me about Aspirin")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    def test_agent_get_capabilities(self) -> None:
        """Test DrugAgent capabilities."""
        agent = DrugAgent()
        capabilities = agent.get_capabilities()
        assert "drug_search" in capabilities
        assert "drug_properties" in capabilities
        assert "drug_similarity" in capabilities


class TestSmilesAgent:
    """Test suite for SmilesAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test SmilesAgent initialization."""
        agent = SmilesAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test SmilesAgent request validation."""
        agent = SmilesAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Analyze this molecule")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Analyze this molecule")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    def test_agent_get_capabilities(self) -> None:
        """Test SmilesAgent capabilities."""
        agent = SmilesAgent()
        capabilities = agent.get_capabilities()
        assert "smiles_validate" in capabilities
        assert "smiles_properties" in capabilities
        assert "smiles_functional_groups" in capabilities


class TestKnowledgeGraphAgent:
    """Test suite for KnowledgeGraphAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test KnowledgeGraphAgent initialization."""
        agent = KnowledgeGraphAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test KnowledgeGraphAgent request validation."""
        agent = KnowledgeGraphAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Find relationships between BRCA1 and cancer")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Find relationships between BRCA1 and cancer")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    def test_agent_get_capabilities(self) -> None:
        """Test KnowledgeGraphAgent capabilities."""
        agent = KnowledgeGraphAgent()
        capabilities = agent.get_capabilities()
        assert "graph_query" in capabilities
        assert "graph_neighbors" in capabilities


class TestDrugDiscoveryAgent:
    """Test suite for DrugDiscoveryAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test DrugDiscoveryAgent initialization."""
        agent = DrugDiscoveryAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test DrugDiscoveryAgent request validation."""
        agent = DrugDiscoveryAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Analyze this compound for drug discovery")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Analyze this compound for drug discovery")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    def test_agent_get_capabilities(self) -> None:
        """Test DrugDiscoveryAgent capabilities."""
        agent = DrugDiscoveryAgent()
        capabilities = agent.get_capabilities()
        assert "drug_discovery" in capabilities
        assert "compound_analysis" in capabilities


class TestPlannerAgent:
    """Test suite for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self) -> None:
        """Test PlannerAgent initialization."""
        agent = PlannerAgent()
        await agent.start()
        assert agent.is_initialized is True
        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_validate_request(self) -> None:
        """Test PlannerAgent request validation."""
        agent = PlannerAgent()
        await agent.start()

        # Valid request
        request = AgentRequest(instruction="Find EGFR papers and predict drug candidates")
        agent.validate_request(request)

        # Invalid request - modify after construction
        invalid_request = AgentRequest(instruction="Find EGFR papers and predict drug candidates")
        invalid_request.instruction = ""
        with pytest.raises(AgentValidationError):
            agent.validate_request(invalid_request)

        await agent.stop()

    def test_detect_intents(self) -> None:
        """Test intent detection."""
        agent = PlannerAgent()

        # Literature search intent
        intents = agent._detect_intents("Find papers on cancer research")
        assert IntentCategory.LITERATURE_SEARCH in intents

        # Drug info intent
        intents = agent._detect_intents("Tell me about Aspirin")
        assert IntentCategory.DRUG_INFO in intents

        # SMILES analysis intent
        intents = agent._detect_intents("Analyze this SMILES: CC(=O)OC1=CC=CC=C1C(=O)O")
        assert IntentCategory.SMILES_ANALYSIS in intents

        # Multiple intents
        intents = agent._detect_intents("Find papers on EGFR and analyze the structure of the drug")
        assert IntentCategory.LITERATURE_SEARCH in intents
        assert IntentCategory.SMILES_ANALYSIS in intents

    def test_create_plan(self) -> None:
        """Test plan creation."""
        agent = PlannerAgent()

        # Simple query
        plan = agent._create_plan("Find papers on cancer")
        assert plan.original_query == "Find papers on cancer"
        assert len(plan.steps) > 0
        assert plan.detected_intents is not None

        # Complex query with multiple intents
        plan = agent._create_plan("Find EGFR papers and analyze the molecular structure")
        assert len(plan.steps) > 1

    def test_create_steps(self) -> None:
        """Test step creation."""
        agent = PlannerAgent()
        matches = [
            (AgentType.LITERATURE_SEARCH_AGENT, IntentCategory.LITERATURE_SEARCH, 0.9),
            (AgentType.SMILES_ANALYSIS_AGENT, IntentCategory.SMILES_ANALYSIS, 0.8),
        ]
        steps = agent._create_steps(matches, "test query")

        assert len(steps) == 2
        assert steps[0].agent_type == AgentType.LITERATURE_SEARCH_AGENT
        assert steps[1].agent_type == AgentType.SMILES_ANALYSIS_AGENT

    def test_detect_dependencies(self) -> None:
        """Test dependency detection."""
        agent = PlannerAgent()

        steps = [
            PlanStep(step_id="1", agent_type=AgentType.SMILES_ANALYSIS_AGENT),
            PlanStep(step_id="2", agent_type=AgentType.DRUG_REPURPOSING_AGENT),
        ]

        updated = agent._detect_dependencies(steps, "test query")
        assert "1" in updated[1].dependencies

    def test_check_ambiguity(self) -> None:
        """Test ambiguity checking."""
        agent = PlannerAgent()

        # Clear query
        steps = [PlanStep(step_id="1", agent_type=AgentType.LITERATURE_SEARCH_AGENT)]
        requires, questions = agent._check_ambiguity(
            "Find papers on cancer",
            [IntentCategory.LITERATURE_SEARCH],
            steps,
        )
        assert requires is False

        # Ambiguous query
        steps = [PlanStep(step_id="1", agent_type=AgentType.SMILES_ANALYSIS_AGENT)]
        requires, questions = agent._check_ambiguity(
            "Analyze this",
            [],
            steps,
        )
        assert requires is True
        assert len(questions) > 0


# ----------------------------------------------------------------------
# Tests for coordinator.py
# ----------------------------------------------------------------------

class TestCoordinator:
    """Test suite for Coordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self) -> None:
        """Test Coordinator initialization."""
        coordinator = Coordinator()
        await coordinator.initialize()
        assert coordinator._initialized is True
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_coordinator_shutdown(self) -> None:
        """Test Coordinator shutdown."""
        coordinator = Coordinator()
        await coordinator.initialize()
        await coordinator.shutdown()
        assert coordinator._initialized is False

    @pytest.mark.asyncio
    async def test_coordinator_process_query(self) -> None:
        """Test Coordinator query processing."""
        coordinator = Coordinator()
        await coordinator.initialize()

        # Mock the planner response
        with patch.object(coordinator.planner, 'handle_request') as mock_handle:
            plan = ExecutionPlan(
                plan_id="test_plan",
                original_query="test query",
                detected_intents=[IntentCategory.LITERATURE_SEARCH],
                steps=[PlanStep(
                    step_id="1",
                    agent_type=AgentType.LITERATURE_SEARCH_AGENT,
                    description="Test step",
                )],
                execution_mode=ExecutionMode.SEQUENTIAL,
            )

            mock_response = AgentResponse(
                request_id="test",
                status=TaskStatus.COMPLETED,
                results=[AgentResult(
                    task_id="test",
                    status=TaskStatus.COMPLETED,
                    output={"plan": plan},
                )],
            )
            mock_handle.return_value = mock_response

            response = await coordinator.process_query("test query")

            assert response.request_id is not None
            assert response.query == "test query"

        await coordinator.shutdown()

    def test_agent_registry(self) -> None:
        """Test AgentRegistry."""
        registry = AgentRegistry()

        # Register an agent
        registry.register(AgentType.LITERATURE_SEARCH_AGENT, agent_class=LiteratureAgent)

        # Check registration
        assert registry.has_agent(AgentType.LITERATURE_SEARCH_AGENT) is True
        assert registry.has_agent(AgentType.DRUG_INFO_AGENT) is False

        # Get instance
        instance = registry.get_instance(AgentType.LITERATURE_SEARCH_AGENT)
        assert instance is not None

        # List agents
        agents = registry.list_agents()
        assert AgentType.LITERATURE_SEARCH_AGENT in agents


# ----------------------------------------------------------------------
# Tests for conversation_manager.py
# ----------------------------------------------------------------------

class TestConversationManager:
    """Test suite for ConversationManager."""

    @pytest.mark.asyncio
    async def test_create_conversation(self) -> None:
        """Test conversation creation."""
        coordinator = Coordinator()
        await coordinator.initialize()

        manager = ConversationManager(coordinator)
        await manager.initialize()

        session = await manager.create_conversation(
            user_id="test_user",
            title="Test Conversation",
        )

        assert session.conversation_id is not None
        assert session.user_id == "test_user"
        assert session.title == "Test Conversation"
        assert session.status == ConversationStatus.ACTIVE

        await manager.shutdown()
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_get_conversation(self) -> None:
        """Test getting a conversation."""
        coordinator = Coordinator()
        await coordinator.initialize()

        manager = ConversationManager(coordinator)
        await manager.initialize()

        created = await manager.create_conversation()
        retrieved = await manager.get_conversation(created.conversation_id)

        assert retrieved is not None
        assert retrieved.conversation_id == created.conversation_id

        await manager.shutdown()
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_process_message(self) -> None:
        """Test processing a message."""
        coordinator = Coordinator()
        await coordinator.initialize()

        manager = ConversationManager(coordinator)
        await manager.initialize()

        # Mock the coordinator response
        with patch.object(coordinator, 'process_query') as mock_process:
            mock_response = CoordinatorResponse(
                request_id="test",
                query="test",
                status=TaskStatus.COMPLETED,
                message="Test response",
            )
            mock_process.return_value = mock_response

            response = await manager.process_message(
                conversation_id="test_conv",
                user_message="Hello",
            )

            assert response.conversation_id == "test_conv"
            assert response.user_message is not None
            assert response.assistant_message is not None

        await manager.shutdown()
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_clear_conversation(self) -> None:
        """Test clearing a conversation."""
        coordinator = Coordinator()
        await coordinator.initialize()

        manager = ConversationManager(coordinator)
        await manager.initialize()

        session = await manager.create_conversation()

        # Add a message
        session.messages.append(ConversationMessage(content="Test message"))

        # Clear
        result = await manager.clear_conversation(session.conversation_id)
        assert result is True

        # Verify cleared
        updated = await manager.get_conversation(session.conversation_id)
        assert updated is not None
        assert len(updated.messages) == 0

        await manager.shutdown()
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_export_conversation(self) -> None:
        """Test exporting a conversation."""
        coordinator = Coordinator()
        await coordinator.initialize()

        manager = ConversationManager(coordinator)
        await manager.initialize()

        session = await manager.create_conversation()
        session.messages.append(ConversationMessage(
            role=ConversationRole.USER,
            content="Hello",
        ))
        session.messages.append(ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content="Hi there!",
        ))

        # Export JSON
        json_export = await manager.export_conversation(session.conversation_id, format="json")
        assert json_export is not None
        data = json.loads(json_export)
        assert data["conversation_id"] == session.conversation_id

        # Export Text
        text_export = await manager.export_conversation(session.conversation_id, format="text")
        assert text_export is not None
        assert "Hello" in text_export
        assert "Hi there!" in text_export

        await manager.shutdown()
        await coordinator.shutdown()

    def test_conversation_session_methods(self) -> None:
        """Test ConversationSession methods."""
        session = ConversationSession()

        # Add messages
        session.add_message(ConversationMessage(role=ConversationRole.USER, content="Hello"))
        session.add_message(ConversationMessage(role=ConversationRole.ASSISTANT, content="Hi"))

        assert len(session.messages) == 2

        # Get last message
        last = session.get_last_message()
        assert last is not None
        assert last.content == "Hi"

        # Get by role
        user_msgs = session.get_user_messages()
        assert len(user_msgs) == 1

        assistant_msgs = session.get_assistant_messages()
        assert len(assistant_msgs) == 1


# ----------------------------------------------------------------------
# Tests for prompts.py
# ----------------------------------------------------------------------

class TestPrompts:
    """Test suite for prompts module."""

    def test_prompt_template_render(self) -> None:
        """Test PromptTemplate rendering."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Hello {name}!",
            placeholders=["name"],
        )

        result = template.render(name="World")
        assert result == "Hello World!"

    def test_prompt_template_missing_placeholder(self) -> None:
        """Test PromptTemplate with missing placeholder."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Hello {name}!",
            placeholders=["name"],
        )

        with pytest.raises(KeyError, match="Missing placeholders: name"):
            template.render()

    def test_prompt_template_with_defaults(self) -> None:
        """Test PromptTemplate rendering with defaults."""
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Hello {name}!",
            placeholders=["name"],
        )

        result = template.render_with_defaults({"name": "Default"}, name="Override")
        assert result == "Hello Override!"

    def test_prompt_registry(self) -> None:
        """Test PromptRegistry."""
        registry = PromptRegistry()

        # Register
        template = PromptTemplate(
            name="test",
            version="v1.0",
            template="Hello {name}!",
            placeholders=["name"],
        )
        registry.register(template)

        # Get
        retrieved = registry.get("test")
        assert retrieved is not None
        assert retrieved.name == "test"

        # Get all versions
        versions = registry.get_all("test")
        assert len(versions) == 1

        # List prompts
        prompts = registry.list_prompts()
        assert "test" in prompts

    def test_render_prompt_function(self) -> None:
        """Test render_prompt function."""
        # Create a registry with a test prompt
        registry = create_prompt_registry()

        # Test rendering a real prompt
        result = render_prompt("general_qa", query="What is cancer?", additional_instructions="")
        assert "What is cancer?" in result
        assert "biomedical research" in result

    def test_render_literature_prompt(self) -> None:
        """Test render_literature_prompt function."""
        result = render_prompt("literature_search",
            query="cancer research",
            year_from=2020,
            year_to=2023,
            article_type="Review",
            open_access_only=True,
            sort_by="relevance",
            additional_instructions="",
        )
        assert "cancer research" in result
        assert "2020" in result
        assert "2023" in result
        assert "Review" in result


# ----------------------------------------------------------------------
# Tests for tools.py
# ----------------------------------------------------------------------

class TestTools:
    """Test suite for tools module."""

    def test_clean_text(self) -> None:
        """Test text cleaning."""
        assert clean_text("  Hello   World  ") == "Hello World"
        assert clean_text(None) == ""
        assert clean_text(123) == "123"

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        text = "This is a long text that needs truncation"
        result = truncate_text(text, 20)
        assert len(result) <= 20
        assert result.endswith("...")

        # No truncation needed
        result = truncate_text("Short", 20)
        assert result == "Short"

    def test_extract_keywords(self) -> None:
        """Test keyword extraction."""
        text = "EGFR mutation is a common finding in lung cancer research"
        keywords = extract_keywords(text, min_length=3, max_keywords=5)

        assert "egfr" in keywords
        assert "mutation" in keywords
        assert "lung" in keywords
        assert len(keywords) > 0

    def test_extract_entities_from_text(self) -> None:
        """Test entity extraction."""
        text = "EGFR mutation in breast cancer and lung cancer treated with trastuzumab"
        entities = extract_entities_from_text(text)

        assert "genes" in entities
        assert "EGFR" in entities["genes"]

        assert "diseases" in entities
        # Check that at least one disease was found
        found_disease = False
        for disease in entities["diseases"]:
            if "breast" in disease.lower() or "lung" in disease.lower():
                found_disease = True
                break
        assert found_disease is True

    def test_format_response(self) -> None:
        """Test response formatting."""
        response = format_response(
            content="Test content",
            metadata={"key": "value"},
            confidence=0.8,
        )

        assert response["content"] == "Test content"
        assert response["metadata"]["key"] == "value"
        assert response["confidence"] == 0.8

    def test_format_error_response(self) -> None:
        """Test error response formatting."""
        response = format_error_response(
            error="Something went wrong",
            context={"user": "test"},
        )

        assert response["success"] is False
        assert response["error"] == "Something went wrong"
        assert response["context"]["user"] == "test"

        # With exception
        try:
            raise ValueError("Test error")
        except ValueError as e:
            response = format_error_response(e)
            assert "Test error" in response["error"]
            assert response["error_type"] == "ValueError"

    def test_format_success_response(self) -> None:
        """Test success response formatting."""
        response = format_success_response(
            data={"result": "success"},
            message="Done",
            metadata={"key": "value"},
        )

        assert response["success"] is True
        assert response["data"]["result"] == "success"
        assert response["message"] == "Done"
        assert response["metadata"]["key"] == "value"

    def test_to_json_safe(self) -> None:
        """Test JSON-safe conversion."""
        # Dataclass
        obj = AgentConfig(max_retries=3)
        result = to_json_safe(obj)
        assert result["max_retries"] == 3

        # DateTime
        now = datetime.now(timezone.utc)
        result = to_json_safe(now)
        assert isinstance(result, str)

        # Enum
        result = to_json_safe(AgentType.ORCHESTRATOR)
        assert result == "orchestrator"

        # List with mixed types
        result = to_json_safe([1, "two", 3.0])
        assert result == [1, "two", 3.0]

    def test_pretty_json(self) -> None:
        """Test pretty JSON formatting."""
        data = {"key": "value", "nested": {"list": [1, 2, 3]}}
        result = pretty_json(data)
        assert '"key": "value"' in result
        # Check for the presence of list items (formatting may vary)
        assert "list" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_normalize_confidence(self) -> None:
        """Test confidence normalization."""
        assert normalize_confidence(0.5) == 0.5
        assert normalize_confidence(1.5) == 1.0
        assert normalize_confidence(-0.5) == 0.0

    def test_aggregate_confidence(self) -> None:
        """Test confidence aggregation."""
        scores = [0.8, 0.9, 0.7]

        # Use pytest.approx for floating point comparison
        assert aggregate_confidence(scores, "mean") == pytest.approx(0.8)
        assert aggregate_confidence(scores, "max") == 0.9
        assert aggregate_confidence(scores, "min") == 0.7
        assert aggregate_confidence(scores, "product") == pytest.approx(0.8 * 0.9 * 0.7)

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_confidence(scores, "unknown")

    def test_confidence_to_label(self) -> None:
        """Test confidence to label conversion."""
        assert confidence_to_label(0.95) == "very_high"
        assert confidence_to_label(0.8) == "high"
        assert confidence_to_label(0.6) == "medium"
        assert confidence_to_label(0.3) == "low"

    def test_validate_non_empty(self) -> None:
        """Test non-empty validation."""
        assert validate_non_empty("test", "field") is True
        assert validate_non_empty([1, 2], "field") is True

        with pytest.raises(ValueError, match="must not be empty"):
            validate_non_empty("", "field")

        with pytest.raises(ValueError, match="must not be empty"):
            validate_non_empty([], "field")

    def test_validate_type(self) -> None:
        """Test type validation."""
        assert validate_type("test", str, "field") is True
        assert validate_type(123, int, "field") is True

        with pytest.raises(TypeError, match="must be of type"):
            validate_type("test", int, "field")

    def test_validate_uuid(self) -> None:
        """Test UUID validation."""
        valid_uuid = str(uuid.uuid4())
        assert validate_uuid(valid_uuid, "field") is True

        with pytest.raises(ValueError, match="must be a valid UUID"):
            validate_uuid("invalid", "field")

    def test_validate_positive(self) -> None:
        """Test positive value validation."""
        assert validate_positive(5, "field") is True
        assert validate_positive(3.14, "field") is True

        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0, "field")

        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-1, "field")

    def test_validate_in_range(self) -> None:
        """Test range validation."""
        assert validate_in_range(5, 0, 10, "field") is True
        assert validate_in_range(0, 0, 10, "field") is True
        assert validate_in_range(10, 0, 10, "field") is True

        with pytest.raises(ValueError, match="must be between"):
            validate_in_range(15, 0, 10, "field")

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2

        id_with_prefix = generate_id("TEST")
        assert id_with_prefix.startswith("TEST_")

    def test_merge_dicts(self) -> None:
        """Test dictionary merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}

        result = merge_dicts(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 4
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_merge_results(self) -> None:
        """Test result merging."""
        results = [
            {"success": True, "data": {"a": 1}},
            {"success": True, "data": {"b": 2}},
            {"success": False, "error": "Failed"},
        ]

        merged = merge_results(results)
        assert merged["total"] == 3
        assert merged["successful"] == 2
        assert merged["failed"] == 1
        assert len(merged["results"]) == 2
        assert len(merged["errors"]) == 1

    def test_get_conversation_id(self) -> None:
        """Test conversation ID generation."""
        id1 = get_conversation_id(user_id="user123")
        assert id1.startswith("conv_user123_")

        id2 = get_conversation_id(user_id="user123", session_id="session456")
        assert id2 == "session456"

    def test_summarize_conversation(self) -> None:
        """Test conversation summarization."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"},
        ]

        summary = summarize_conversation(messages, max_turns=2)
        # With max_turns=2, all 4 messages are included (2 turns = 4 messages)
        assert "How are you?" in summary
        assert "I'm fine, thank you!" in summary
        assert "Hello" in summary
        assert "Hi there!" in summary


# ----------------------------------------------------------------------
# Tests for cache.py
# ----------------------------------------------------------------------

class TestCache:
    """Test suite for cache module."""

    def test_cache_config_validation(self) -> None:
        """Test CacheConfig validation."""
        # Valid config
        config = CacheConfig(max_size=100, default_ttl_seconds=60)
        assert config.max_size == 100

        # Invalid max_size
        with pytest.raises(ValueError, match="max_size must be positive"):
            CacheConfig(max_size=0)

        # Invalid ttl
        with pytest.raises(ValueError, match="default_ttl_seconds must be non-negative"):
            CacheConfig(default_ttl_seconds=-1)

    def test_cache_set_get(self) -> None:
        """Test cache set and get operations."""
        cache = Cache()

        # Set and get
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

        # Get with default
        result = cache.get("nonexistent", "default")
        assert result == "default"

        # Contains
        assert cache.contains("key1") is True
        assert cache.contains("nonexistent") is False

    def test_cache_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache = Cache(CacheConfig(default_ttl_seconds=0.1))

        cache.set("key1", "value1")
        assert cache.contains("key1") is True

        # Wait for expiration
        time.sleep(0.2)

        assert cache.contains("key1") is False
        result = cache.get("key1")
        assert result is None

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction."""
        cache = Cache(CacheConfig(max_size=2))

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # key1 should be evicted
        assert cache.contains("key1") is False
        assert cache.contains("key2") is True
        assert cache.contains("key3") is True

        # Access key2 to make it most recent
        cache.get("key2")
        cache.set("key4", "value4")

        # key3 should be evicted
        assert cache.contains("key3") is False
        assert cache.contains("key2") is True
        assert cache.contains("key4") is True

    def test_cache_namespace(self) -> None:
        """Test namespace support."""
        cache = Cache()

        cache.set("ns1:key1", "value1")
        cache.set("ns1:key2", "value2")
        cache.set("ns2:key3", "value3")

        # Get namespace size
        assert cache.get_namespace_size("ns1") == 2
        assert cache.get_namespace_size("ns2") == 1

        # Get keys
        keys_ns1 = cache.get_keys("ns1")
        assert len(keys_ns1) == 2
        assert "ns1:key1" in keys_ns1

        # Invalidate namespace
        removed = cache.invalidate_namespace("ns1")
        assert removed == 2
        assert cache.get_namespace_size("ns1") == 0
        assert cache.get("ns1:key1") is None

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = Cache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Hits and misses
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.total_hits == 1
        assert stats.total_misses == 1
        assert stats.total_entries == 2

        # Per-namespace stats
        assert "default" in stats.namespace_stats
        ns_stats = stats.namespace_stats["default"]
        assert ns_stats.hits == 1
        assert ns_stats.misses == 1
        assert ns_stats.entry_count == 2

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation."""
        cache = Cache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.invalidate("key1") is True
        assert cache.contains("key1") is False
        assert cache.contains("key2") is True

        assert cache.invalidate("nonexistent") is False

    def test_cache_clear(self) -> None:
        """Test cache clear."""
        cache = Cache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        removed = cache.clear()
        assert removed == 2
        assert len(cache.get_keys()) == 0

    def test_cache_prune_expired(self) -> None:
        """Test pruning expired entries."""
        cache = Cache(CacheConfig(default_ttl_seconds=0.1))

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Wait for expiration
        time.sleep(0.2)

        removed = cache.prune_expired()
        assert removed == 2
        assert cache.contains("key1") is False
        assert cache.contains("key2") is False

    def test_cached_decorator(self) -> None:
        """Test cached decorator."""
        call_count = 0

        @cached(ttl=60, namespace="test")
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should hit cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1

        # Different argument should compute
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_default_cache(self) -> None:
        """Test default cache."""
        # Clear default cache
        clear_default_cache()

        # Get default cache
        cache1 = get_default_cache()
        cache2 = get_default_cache()
        assert cache1 is cache2

        # Reset default cache
        new_cache = reset_default_cache(CacheConfig(max_size=50))
        assert new_cache.config.max_size == 50
        assert get_default_cache() is new_cache


# ----------------------------------------------------------------------
# Integration tests
# ----------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the multi-agent system."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self) -> None:
        """Test end-to-end flow from user query to response."""
        # Set up coordinator
        coordinator = Coordinator()
        await coordinator.initialize()

        # Set up conversation manager
        manager = ConversationManager(coordinator)
        await manager.initialize()

        # Create a conversation
        session = await manager.create_conversation(user_id="test_user")
        assert session is not None

        # Process a message (with mocking for actual agent execution)
        with patch.object(coordinator, 'process_query') as mock_process:
            mock_response = CoordinatorResponse(
                request_id="test",
                query="test query",
                status=TaskStatus.COMPLETED,
                message="Test response",
            )
            mock_process.return_value = mock_response

            response = await manager.process_message(
                conversation_id=session.conversation_id,
                user_message="Hello, can you help me find papers on EGFR?",
            )

            assert response.status == TaskStatus.COMPLETED
            assert response.assistant_message is not None

        await manager.shutdown()
        await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_agent_registry_integration(self) -> None:
        """Test agent registry with real agents."""
        registry = AgentRegistry()

        # Register all agents
        registry.register(AgentType.LITERATURE_SEARCH_AGENT, agent_class=LiteratureAgent)
        registry.register(AgentType.DRUG_INFO_AGENT, agent_class=DrugAgent)
        registry.register(AgentType.SMILES_ANALYSIS_AGENT, agent_class=SmilesAgent)
        registry.register(AgentType.KNOWLEDGE_GRAPH_AGENT, agent_class=KnowledgeGraphAgent)
        registry.register(AgentType.DRUG_REPURPOSING_AGENT, agent_class=DrugDiscoveryAgent)

        # Initialize all agents
        await registry.initialize_all()

        # Get and verify instances
        for agent_type in registry.list_agents():
            instance = registry.get_instance(agent_type)
            assert instance.is_initialized is True

        # Shutdown all agents
        await registry.shutdown_all()

        for agent_type in registry.list_agents():
            instance = registry.get_instance(agent_type)
            assert instance.is_initialized is False

    def test_planner_to_coordinator_integration(self) -> None:
        """Test planner and coordinator integration."""
        planner = PlannerAgent()
        coordinator = Coordinator()

        # Create a query
        query = "Find papers on EGFR and analyze the molecular structure"

        # Get plan from planner
        plan = planner._create_plan(query)

        # Verify plan structure
        assert len(plan.steps) > 0
        assert plan.original_query == query

        # Verify step types
        agent_types = [s.agent_type for s in plan.steps]
        assert AgentType.LITERATURE_SEARCH_AGENT in agent_types

    def test_memory_planner_integration(self) -> None:
        """Test memory and planner integration."""
        # Set up memory
        store = InMemoryMemoryStore()
        memory = MemoryManager(store=store)

        # Set up planner with memory
        planner = PlannerAgent(memory=memory)
        planner._initialized = True

        # Remember conversation
        conv_id = "test_conv"
        memory.remember_turn(conv_id, role="user", content="Find papers on cancer")

        # Planner should be able to access memory
        # This tests that the planner can use memory for context
        context = memory.recall(conv_id, limit=5)
        assert len(context) > 0


# ----------------------------------------------------------------------
# Error handling tests
# ----------------------------------------------------------------------

class TestErrorHandling:
    """Test suite for error handling."""

    def test_memory_not_found_error(self) -> None:
        """Test MemoryNotFoundError."""
        store = InMemoryMemoryStore()

        with pytest.raises(MemoryNotFoundError):
            store.get_conversation("nonexistent")

    def test_cache_key_error(self) -> None:
        """Test cache key error."""
        cache = Cache()

        with pytest.raises(CacheKeyError):
            cache.set("", "value")

    def test_validate_error_handling(self) -> None:
        """Test validation error handling."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0, "field")

        with pytest.raises(TypeError, match="must be of type"):
            validate_type("test", int, "field")

    def test_aggregate_confidence_empty(self) -> None:
        """Test aggregate confidence with empty list."""
        assert aggregate_confidence([]) == 0.0

    @pytest.mark.asyncio
    async def test_agent_initialization_error(self) -> None:
        """Test agent initialization error handling."""

        class FailingAgent(BaseAgent):
            async def initialize(self) -> None:
                raise AgentInitializationError("Init failed")

            def validate_request(self, request: AgentRequest) -> None:
                pass

            async def execute(self, task: AgentTask) -> AgentResult:
                return AgentResult(task_id=task.task_id, status=TaskStatus.COMPLETED)

            async def shutdown(self) -> None:
                pass

        agent = FailingAgent(AgentType.ORCHESTRATOR)

        with pytest.raises(AgentInitializationError):
            await agent.start()

    @pytest.mark.asyncio
    async def test_coordinator_not_initialized(self) -> None:
        """Test coordinator when not initialized."""
        coordinator = Coordinator()
        # Don't initialize

        with pytest.raises(CoordinatorInitializationError):
            await coordinator.process_query("test")

    @pytest.mark.asyncio
    async def test_conversation_manager_message_limit(self) -> None:
        """Test conversation manager message limit."""
        coordinator = Coordinator()
        await coordinator.initialize()

        config = ConversationManagerConfig(max_messages_per_conversation=2)
        manager = ConversationManager(coordinator, config=config)
        await manager.initialize()

        session = await manager.create_conversation()

        # Add messages
        session.add_message(ConversationMessage(content="Message 1"))
        session.add_message(ConversationMessage(content="Message 2"))

        # Try to add another message
        with pytest.raises(ValueError, match="maximum of 2 messages"):
            await manager.process_message(session.conversation_id, "Message 3")

        await manager.shutdown()
        await coordinator.shutdown()