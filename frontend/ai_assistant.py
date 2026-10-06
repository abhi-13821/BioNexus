"""
frontend/ai_assistant.py

Streamlit frontend for the BioNexus Multi-Agent AI Assistant.
Uses SmartCoordinator with Ollama for intelligent conversations.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict

import streamlit as st

# Import agent framework components
try:
    from agents.conversation_manager import (
        ConversationManager,
        ConversationMessage,
        ConversationResponse,
        ConversationRole,
        ConversationSession,
        ConversationStatus,
    )
    from agents.coordinator import Coordinator, CoordinatorResponse
    from agents.models import AgentType, ConfidenceScore, TaskStatus
    from agents.smart_coordinator import SmartCoordinator
    from agents.coordinator import AgentRegistry
except ImportError as e:
    st.error(f"❌ Failed to import agent framework: {e}")
    st.error("Make sure all agent files are in the 'agents/' folder")
    st.stop()

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="BioNexus AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------

st.markdown("""
<style>
    .main { padding: 0 1rem; }
    
    .ai-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    
    .ai-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .ai-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.25rem 0 0 0;
    }
    
    .chat-message-user {
        background: #2d3748;
        color: #f7fafc;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        margin-left: auto;
        max-width: 85%;
        border-bottom-right-radius: 4px;
    }
    
    .chat-message-assistant {
        background: #1a202c;
        color: #e2e8f0;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        margin-right: auto;
        max-width: 85%;
        border-bottom-left-radius: 4px;
        border: 1px solid #2d3748;
    }
    
    .chat-message .timestamp {
        font-size: 0.7rem;
        opacity: 0.5;
        margin-top: 0.3rem;
        text-align: right;
    }
    
    .metadata-container {
        background: #0d1117;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
    }
    
    .badge-success { background: rgba(72, 187, 120, 0.2); color: #48bb78; }
    .badge-failed { background: rgba(252, 129, 129, 0.2); color: #fc8181; }
    .badge-agent { background: rgba(66, 153, 225, 0.2); color: #63b3ed; }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: #0d1117;
        padding: 0.8rem 0;
        border-top: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "ai_initialized": False,
        "ai_conversation_manager": None,
        "ai_current_session_id": None,
        "ai_messages": [],
        "ai_is_processing": False,
        "ai_show_metadata": True,
        "ai_last_papers": [],
        "ai_last_query": "",
        "ai_last_results": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ----------------------------------------------------------------------
# Agent initialization - FIXED with SmartCoordinator
# ----------------------------------------------------------------------

@st.cache_resource
def get_conversation_manager() -> ConversationManager:
    """
    Create and cache the ConversationManager with SmartCoordinator.
    """
    try:
        # Import agent classes
        from agents.literature_agent import LiteratureAgent
        from agents.drug_agent import DrugAgent
        from agents.smiles_agent import SmilesAgent
        from agents.knowledge_graph_agent import KnowledgeGraphAgent
        from agents.drug_discovery_agent import DrugDiscoveryAgent
        
        # Create registry with all agents
        registry = AgentRegistry()
        registry.register(AgentType.LITERATURE_SEARCH_AGENT, agent_class=LiteratureAgent)
        registry.register(AgentType.DRUG_INFO_AGENT, agent_class=DrugAgent)
        registry.register(AgentType.SMILES_ANALYSIS_AGENT, agent_class=SmilesAgent)
        registry.register(AgentType.KNOWLEDGE_GRAPH_AGENT, agent_class=KnowledgeGraphAgent)
        registry.register(AgentType.DRUG_REPURPOSING_AGENT, agent_class=DrugDiscoveryAgent)
        
        logger.info("✅ All agents registered successfully")
        
        # Create SmartCoordinator with Ollama
        coordinator = SmartCoordinator(
            registry=registry,
            model="mistral:7b",
            ollama_url="http://localhost:11434/api/generate",
        )
        
        # Initialize
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coordinator.initialize())
        
        # Create conversation manager
        manager = ConversationManager(coordinator=coordinator)
        loop.run_until_complete(manager.initialize())
        
        logger.info("✅ SmartCoordinator with mistral:7b initialized")
        return manager
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        st.error(f"❌ Missing module: {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Failed: {e}")
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()


# ----------------------------------------------------------------------
# Follow-up question handler
# ----------------------------------------------------------------------

def _handle_follow_up_question(query: str) -> Optional[str]:
    """
    Handle follow-up questions about previous results.
    """
    papers = st.session_state.get("ai_last_papers", [])
    
    if not papers:
        return None
    
    query_lower = query.lower()
    
    # Check if asking about a specific paper by number
    match = re.search(r'paper\s*#?\s*(\d+)|#\s*(\d+)', query_lower)
    if match:
        idx = int(match.group(1) or match.group(2))
        if 1 <= idx <= len(papers):
            paper = papers[idx - 1]
            authors = paper.get("authors", [])
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            
            response = f"## 📄 Paper #{idx}\n\n"
            response += f"**Title:** {paper.get('title', 'Untitled')}\n\n"
            response += f"**Authors:** {author_str}\n\n"
            response += f"**Year:** {paper.get('year', 'n.d.')}\n\n"
            response += f"**Journal:** {paper.get('journal', 'Unknown Journal')}\n\n"
            response += f"**DOI:** {paper.get('doi', 'N/A')}\n\n"
            response += f"**PMID:** {paper.get('pmid', 'N/A')}\n\n"
            if paper.get('abstract'):
                response += f"**Abstract:**\n{paper.get('abstract')[:1000]}..."
            
            return response
    
    # Check if asking for summary
    if any(word in query_lower for word in ["summarize", "summary", "overview"]):
        if papers:
            response = f"## 📚 Overview of {len(papers)} Papers\n\n"
            response += f"Found {len(papers)} papers related to **'{st.session_state.get('ai_last_query', 'your query')}'**.\n\n"
            response += "**Key papers:**\n\n"
            for i, paper in enumerate(papers[:5], 1):
                title = paper.get('title', 'Untitled')
                year = paper.get('year', 'n.d.')
                journal = paper.get('journal', 'Unknown Journal')
                response += f"**{i}.** {title}\n"
                response += f"   ({year}) - {journal}\n\n"
            
            if len(papers) > 5:
                response += f"\n*... and {len(papers) - 5} more papers*\n\n"
            
            response += "To see details of a specific paper, ask 'explain paper #' (e.g., 'explain paper 3')."
            return response
    
    return None


# ----------------------------------------------------------------------
# Message display
# ----------------------------------------------------------------------

def render_chat_interface() -> None:
    """Render the main chat interface."""
    for msg in st.session_state.ai_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        metadata = msg.get("metadata")
        timestamp = msg.get("timestamp", "")
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
                if timestamp:
                    st.caption(f"{timestamp}")
        
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
                if metadata and st.session_state.ai_show_metadata:
                    with st.expander("📊 Execution Details"):
                        _render_metadata(metadata)
                if timestamp:
                    st.caption(f"{timestamp}")
        
        elif role == "error":
            with st.chat_message("assistant"):
                st.error(content)


def _render_metadata(metadata: dict[str, Any]) -> None:
    """Render execution metadata."""
    status = metadata.get("status", "unknown")
    status_class = "badge-success" if status == "completed" else "badge-failed"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Status:** <span class='{status_class}'>{status.upper()}</span>", unsafe_allow_html=True)
    
    with col2:
        confidence = metadata.get("confidence")
        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence:.2f}")
    
    with col3:
        duration = metadata.get("duration_ms")
        if duration is not None:
            st.markdown(f"**Duration:** {duration:.0f}ms")
    
    agents = metadata.get("agents_invoked", [])
    if agents:
        st.markdown("**Agents Used:**")
        agent_tags = " ".join(f"<span class='badge-agent'>{a}</span>" for a in agents)
        st.markdown(agent_tags, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("### 🤖 AI Assistant")
        st.markdown("---")
        
        message_count = len(st.session_state.ai_messages)
        st.markdown(f"**Messages:** {message_count}")
        
        papers = st.session_state.get("ai_last_papers", [])
        if papers:
            st.markdown(f"**Cached Papers:** {len(papers)}")
            st.markdown(f"**Last Query:** {st.session_state.get('ai_last_query', '')[:30]}...")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        st.session_state.ai_show_metadata = st.checkbox(
            "Show execution details",
            value=st.session_state.ai_show_metadata,
        )
        
        st.markdown("---")
        st.markdown("### 🛠️ Actions")
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.ai_messages = []
            st.session_state.ai_last_papers = []
            st.session_state.ai_last_query = ""
            st.session_state.ai_last_results = {}
            st.rerun()
        
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.ai_messages = []
            st.session_state.ai_last_papers = []
            st.session_state.ai_last_query = ""
            st.session_state.ai_last_results = {}
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Stats")
        
        user_msgs = sum(1 for m in st.session_state.ai_messages if m.get("role") == "user")
        assistant_msgs = sum(1 for m in st.session_state.ai_messages if m.get("role") == "assistant")
        
        st.markdown(f"**User messages:** {user_msgs}")
        st.markdown(f"**Assistant messages:** {assistant_msgs}")
        
        if st.session_state.ai_initialized:
            st.success("🟢 System Ready")
        else:
            st.warning("🟡 Initializing...")


# ----------------------------------------------------------------------
# Query processing
# ----------------------------------------------------------------------

def process_user_query(query: str) -> None:
    """
    Process a user query through the ConversationManager.
    """
    if st.session_state.ai_is_processing:
        st.warning("Please wait for the current request to complete.")
        return
    
    if not query or not query.strip():
        return
    
    # Check for follow-up questions about previous results
    follow_up_response = _handle_follow_up_question(query)
    if follow_up_response:
        assistant_message = {
            "role": "assistant",
            "content": follow_up_response,
            "metadata": {"status": "completed", "agents_invoked": ["memory"], "follow_up": True},
            "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
        }
        st.session_state.ai_messages.append(assistant_message)
        st.rerun()
        return
    
    # Add user message
    user_message = {
        "role": "user",
        "content": query,
        "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
    }
    st.session_state.ai_messages.append(user_message)
    
    st.session_state.ai_is_processing = True
    
    try:
        manager = st.session_state.ai_conversation_manager
        if manager is None:
            raise RuntimeError("AI assistant not initialized")
        
        session_id = st.session_state.ai_current_session_id
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(
            manager.process_message(
                conversation_id=session_id or "new_session",
                user_message=query,
                user_id=st.session_state.get("user_id", "streamlit_user"),
            )
        )
        loop.close()
        
        if response.conversation_id:
            st.session_state.ai_current_session_id = response.conversation_id
        
        # Prepare metadata
        metadata = {}
        if response.coordinator_response:
            # Extract papers from response for caching
            papers = []
            if hasattr(response.coordinator_response, 'merged_output'):
                merged = response.coordinator_response.merged_output
                if isinstance(merged, dict) and "papers" in merged:
                    papers = merged.get("papers", [])
                    if papers:
                        st.session_state.ai_last_papers = papers
                        st.session_state.ai_last_query = query
            
            metadata = {
                "status": response.coordinator_response.status.value if hasattr(response.coordinator_response.status, 'value') else "completed",
                "confidence": response.coordinator_response.confidence.value if hasattr(response.coordinator_response, 'confidence') and response.coordinator_response.confidence else None,
                "duration_ms": getattr(response.coordinator_response, 'total_duration_ms', None),
                "agents_invoked": ["SmartCoordinator"],
                "agent_results": [],
                "errors": [],
            }
        
        # Build assistant content
        assistant_content = response.message or "I processed your request."
        
        # Add assistant response
        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
        }
        st.session_state.ai_messages.append(assistant_message)
        
        # Check for errors
        if hasattr(response, 'status') and response.status == TaskStatus.FAILED:
            error_msg = getattr(response, 'error', "An error occurred.")
            error_message = {
                "role": "error",
                "content": f"⚠️ {error_msg}",
                "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
            }
            st.session_state.ai_messages.append(error_message)
        
    except Exception as e:
        logger.exception("Error processing query: %s", e)
        error_message = {
            "role": "error",
            "content": f"❌ Error: {str(e)}",
            "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
        }
        st.session_state.ai_messages.append(error_message)
    
    finally:
        st.session_state.ai_is_processing = False
        st.rerun()


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------

def main() -> None:
    """Main application entry point."""
    
    if "ai_initialized" not in st.session_state:
        st.session_state.ai_initialized = False
    if "ai_conversation_manager" not in st.session_state:
        st.session_state.ai_conversation_manager = None
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    if "ai_is_processing" not in st.session_state:
        st.session_state.ai_is_processing = False
    if "ai_conversation_id" not in st.session_state:
        st.session_state.ai_conversation_id = None
    if "ai_show_metadata" not in st.session_state:
        st.session_state.ai_show_metadata = False
    
    # Header
    st.markdown("""
    <div class="ai-header">
        <h1>🤖 BioNexus AI Assistant</h1>
        <p>Multi-agent AI system for biomedical research and drug discovery</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize manager if not already done
    if not st.session_state.ai_initialized:
        with st.spinner("🧠 Initializing AI Assistant with Ollama..."):
            try:
                manager = get_conversation_manager()
                st.session_state.ai_conversation_manager = manager
                st.session_state.ai_initialized = True
                
                # Add welcome message
                if not st.session_state.ai_messages:
                    welcome_msg = {
                        "role": "assistant",
                        "content": "👋 Welcome to the BioNexus AI Assistant!\n\n"
                        "I'm a multi-agent AI system with **LLM intelligence** (powered by Ollama).\n\n"
                        "**What I can do:**\n\n"
                        "📚 **Literature Search** - Find and analyze biomedical research papers\n"
                        "💊 **Drug Information** - Get detailed drug and compound information\n"
                        "🧪 **SMILES Analysis** - Analyze molecular structures\n"
                        "🕸️ **Knowledge Graph** - Explore biomedical relationships\n"
                        "🔬 **Drug Discovery** - AI-powered drug discovery and candidate evaluation\n\n"
                        "**Try asking:**\n"
                        "- 'Find papers about EGFR mutations'\n"
                        "- 'Explain paper #3'\n"
                        "- 'Summarize the papers'\n"
                        "- 'Tell me about Paracetamol'\n"
                        "- 'Analyze this SMILES: CC(=O)OC1=CC=CC=C1C(=O)O'\n\n"
                        "How can I assist you today?",
                        "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p"),
                    }
                    st.session_state.ai_messages.append(welcome_msg)
                
            except Exception as e:
                st.error(f"❌ Failed to initialize AI Assistant: {e}")
                st.error("Make sure Ollama is running with 'ollama serve'")
                return
    
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat_interface()
    
    # Chat input
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        if st.session_state.ai_is_processing:
            st.info("🧠 Processing with AI...")
        
        query = st.chat_input(
            "Ask me about biomedical research, drugs, compounds, or drug discovery...",
            disabled=st.session_state.ai_is_processing,
        )
        
        if query:
            process_user_query(query)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("BioNexus AI Assistant | Powered by Multi-Agent AI + Ollama (mistral:7b)")