"""
agents/smart_coordinator.py

Smart Coordinator with full LLM integration for conversational agents.
Uses Ollama (Mistral 7B) for intelligent responses and explanations.
"""

from __future__ import annotations

import asyncio
import logging
import re
import requests
from datetime import datetime, timezone
from typing import Any, Optional, Dict, List

from agents.base_agent import AgentType
from agents.coordinator import Coordinator, CoordinatorResponse
from agents.models import AgentRequest, ConfidenceScore, TaskStatus
from agents.tools import generate_id

logger = logging.getLogger(__name__)


class SmartCoordinator(Coordinator):
    """
    Smart Coordinator that uses a local LLM (Ollama) to:
    - Detect user intent
    - Route to appropriate agents
    - Explain and summarize results
    - Handle follow-up questions with context
    """

    def __init__(
        self,
        *,
        config=None,
        registry=None,
        memory=None,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "mistral:7b",
    ):
        super().__init__(config=config, registry=registry, memory=memory)
        self.ollama_url = ollama_url
        self.model = model
        self._ollama_available = self._check_ollama()
        self._initialized = False
        self._cached_papers = []
        self._cached_query = ""
        
        if self._ollama_available:
            logger.info(f"✅ Ollama available with model: {model}")
        else:
            logger.warning("⚠️ Ollama not available. Using fallback mode.")

    def _check_ollama(self) -> bool:
        """Check if Ollama is available using the Python package."""
        try:
            import ollama
            ollama.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

    async def initialize(self) -> None:
        """Initialize the coordinator."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("✅ SmartCoordinator initialized")

    async def _ensure_agent_started(self, agent_type: AgentType):
        """Ensure a specific agent is started."""
        try:
            agent = self.registry.get_instance(agent_type)
            if not agent.is_initialized:
                logger.info(f"Starting agent on demand: {agent_type.value}")
                await agent.start()
                logger.info(f"✅ Agent {agent_type.value} started")
            return agent
        except Exception as e:
            logger.error(f"Failed to start agent {agent_type}: {e}")
            raise

    async def process_query(self, query: str, conversation_id: Optional[str] = None, **kwargs) -> CoordinatorResponse:
        """Process a user query with LLM intelligence."""

        # Step 1: Detect intent
        intent = self._detect_intent(query)
        
        # Step 2: Get context from memory
        context = self._get_context(conversation_id)
        
        # Step 3: Route to appropriate handler
        try:
            intent_type = intent.get("type", "general")
            if intent_type == "literature":
                result = await self._handle_literature(query, context, conversation_id)
            elif intent_type == "drug":
                result = await self._handle_drug(query, context, conversation_id)
            elif intent_type == "smiles":
                result = await self._handle_smiles(query, context, conversation_id)
            elif intent_type == "knowledge_graph":
                result = await self._handle_knowledge_graph(query, context, conversation_id)
            elif intent_type == "drug_discovery":
                result = await self._handle_drug_discovery(query, context, conversation_id)
            elif intent_type == "follow_up":
                result = await self._handle_follow_up(query, context, conversation_id)
            elif intent_type == "compare":
                result = await self._handle_compare(query, context, conversation_id)
            else:
                result = await self._handle_general(query, context, conversation_id)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            result = {"answer": f"⚠️ Error: {str(e)}", "confidence": 0.3}
        
        # Step 4: Store in memory
        if conversation_id:
            try:
                self.memory.remember_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result.get("answer", ""),
                    agent_type=AgentType.ORCHESTRATOR,
                )
            except Exception:
                pass
        
        # Step 5: Return response
        return CoordinatorResponse(
            request_id=generate_id("SMART"),
            query=query,
            status=TaskStatus.COMPLETED,
            message=result.get("answer", "No response generated."),
            merged_output=result,
            confidence=ConfidenceScore(
                value=result.get("confidence", 0.85),
                basis=f"SmartCoordinator processed with {self.model}",
                source_module="smart_coordinator",
            ),
            created_at=datetime.now(timezone.utc),
        )

    def _detect_intent(self, query: str) -> dict:
        """Detect user intent from query."""
        q = query.lower()
        
        # Follow-up detection for specific paper
        if re.search(r'(?:paper|#)\s*(\d+)', q):
            return {"type": "follow_up", "subtype": "paper_detail"}
        if "summarize" in q or "summary" in q or "overview" in q:
            return {"type": "follow_up", "subtype": "summarize"}
        if "compare" in q:
            return {"type": "compare"}
        
        # Literature search
        literature_keywords = ["paper", "papers", "publication", "research", "study", "literature", "find", "search"]
        if any(w in q for w in literature_keywords):
            return {"type": "literature", "query": query}
        
        # Drug info
        drug_keywords = ["drug", "medicine", "medication", "compound", "pharma", "pharmaceutical"]
        if any(w in q for w in drug_keywords):
            return {"type": "drug", "query": query}
        
        # SMILES analysis
        smiles_keywords = ["smiles", "structure", "molecule", "molecular", "chemical"]
        if any(w in q for w in smiles_keywords):
            return {"type": "smiles", "query": query}
        
        # Knowledge graph
        kg_keywords = ["relationship", "connection", "pathway", "gene", "protein", "interaction"]
        if any(w in q for w in kg_keywords):
            return {"type": "knowledge_graph", "query": query}
        
        # Drug discovery
        dd_keywords = ["discovery", "candidate", "similar", "repurpose", "repurposing", "screen"]
        if any(w in q for w in dd_keywords):
            return {"type": "drug_discovery", "query": query}
        
        return {"type": "general", "query": query}

    def _get_context(self, conversation_id: Optional[str]) -> str:
        """Get conversation context from memory."""
        if not conversation_id:
            return ""
        try:
            records = self.memory.recall(conversation_id, limit=10)
            if records:
                return "\n".join(f"{r.role}: {r.content}" for r in records)
        except Exception:
            pass
        return ""

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call Ollama LLM for intelligent responses."""
        if not self._ollama_available:
            return None
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 1500,
                },
                timeout=120,
            )
            if response.status_code == 200:
                result = response.json().get("response")
                if result:
                    return result.strip()
                return None
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out (120s)")
            return None
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

    # ============================================================
    # HANDLERS
    # ============================================================

    async def _handle_literature(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle literature search with LLM explanation."""
        try:
            agent = await self._ensure_agent_started(AgentType.LITERATURE_SEARCH_AGENT)
            
            request = AgentRequest(instruction=query, parameters={"query": query})
            response = await agent.handle_request(request)
            
            papers = []
            if response.results and response.results[0].output:
                output = response.results[0].output
                papers = output.get("papers", [])
                if papers is None:
                    papers = []
                logger.info(f"Literature search found {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"Literature agent error: {e}")
            papers = []
        
        # Ensure papers is always a list
        if papers is None:
            papers = []
        
        formatted_papers = self._format_papers(papers)
        if formatted_papers is None:
            formatted_papers = "📚 No papers found. Please try a different search query."
        
        llm_summary = ""
        if papers and len(papers) > 0:
            prompt = (
                f"Summarize these {len(papers)} research papers about {query[:50]}...:\n\n"
                f"{self._papers_for_llm(papers[:5])}\n\n"
                "Provide:\n"
                "1. A brief overview (2-3 sentences)\n"
                "2. Key themes and findings\n"
                "3. Most interesting/important paper and why\n"
                "4. A simple recommendation for further reading\n\n"
                "Keep it concise and conversational."
            )
            
            llm_summary = await self._call_llm(prompt)
            if llm_summary:
                llm_summary = f"\n\n🤖 **AI Summary:**\n{llm_summary}"
            else:
                llm_summary = ""
        
        self._cached_papers = papers
        self._cached_query = query
        
        answer = formatted_papers + llm_summary
        return {"answer": answer, "papers": papers, "count": len(papers), "confidence": 0.9}

    async def _handle_follow_up(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle follow-up questions about previous results."""
        # Get cached papers from the last search
        papers = getattr(self, '_cached_papers', [])
        
        if not papers:
            return {
                "answer": "I don't have any papers to refer to. Please do a search first (e.g., 'Find papers about EGFR mutations').",
                "confidence": 0.5
            }
        
        # Check if asking about a specific paper
        match = re.search(r'(?:paper|#)\s*(\d+)', query.lower())
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= len(papers):
                paper = papers[idx - 1]
                
                formatted_paper = self._format_paper_details(paper, idx)
                
                prompt = (
                    f"Explain this research paper in simple, clear terms:\n\n"
                    f"Title: {paper.get('title', 'Unknown Title')}\n"
                    f"Authors: {', '.join(paper.get('authors', ['Unknown']))}\n"
                    f"Year: {paper.get('year', 'n.d.')}\n"
                    f"Journal: {paper.get('journal', 'Unknown')}\n"
                    f"Abstract: {paper.get('abstract', 'No abstract available.')[:1500]}\n\n"
                    "Provide:\n"
                    "1. What was the main question or problem?\n"
                    "2. What did they do (methods)?\n"
                    "3. What did they find (key results)?\n"
                    "4. Why does this matter (significance)?\n\n"
                    "Make it easy to understand for a non-expert. Be conversational."
                )
                
                explanation = await self._call_llm(prompt)
                if explanation:
                    return {"answer": f"## 📄 Paper #{idx}\n\n{explanation}", "paper": paper, "confidence": 0.95}
                else:
                    return {"answer": formatted_paper, "paper": paper, "confidence": 0.8}
        
        # Check if asking for summary
        if "summarize" in query.lower() or "summary" in query.lower():
            prompt = (
                f"Summarize these {len(papers)} research papers:\n\n"
                f"{self._papers_for_llm(papers[:10])}\n\n"
                "Provide:\n"
                "1. Overall summary of the research\n"
                "2. Key themes that appear\n"
                "3. Most significant findings\n"
                "4. Gaps or future directions\n\n"
                "Keep it concise and useful."
            )
            
            summary = await self._call_llm(prompt)
            if summary:
                return {"answer": f"## 📚 Research Summary\n\n{summary}", "confidence": 0.9}
        
        return {
            "answer": "I'm not sure how to answer that. Try asking about a specific paper like 'Explain paper 3' or ask me to 'Summarize the papers'.",
            "confidence": 0.5
        }

    async def _handle_drug(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle drug information with LLM explanation."""
        try:
            agent = await self._ensure_agent_started(AgentType.DRUG_INFO_AGENT)
            request = AgentRequest(instruction=query, parameters={"query": query})
            response = await agent.handle_request(request)
            
            drug_data = {}
            if response.results and response.results[0].output:
                drug_data = response.results[0].output
        except Exception as e:
            logger.error(f"Drug agent error: {e}")
            drug_data = {}
        
        formatted_drug = self._format_drug(drug_data)
        if formatted_drug is None:
            formatted_drug = "💊 Drug information not found."
        
        llm_explanation = ""
        if drug_data:
            prompt = (
                f"Explain this drug information in simple, clear terms:\n\n"
                f"{drug_data}\n\n"
                "Focus on:\n"
                "1. What the drug is and what it's used for\n"
                "2. How it works (mechanism of action)\n"
                "3. Important properties\n"
                "4. Anything interesting or unique about it\n\n"
                "Be conversational and helpful."
            )
            
            llm_explanation = await self._call_llm(prompt)
            if llm_explanation:
                llm_explanation = f"\n\n🤖 **AI Explanation:**\n{llm_explanation}"
            else:
                llm_explanation = ""
        
        return {"answer": formatted_drug + llm_explanation, "drug_data": drug_data, "confidence": 0.9}

    async def _handle_smiles(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle SMILES analysis with LLM explanation."""
        pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}'
        smiles_match = re.search(pattern, query)
        
        if not smiles_match:
            return {
                "answer": "🔬 No SMILES string found in your query. Please provide a SMILES string.",
                "confidence": 0.5
            }
        
        smiles = smiles_match.group(0)
        
        try:
            agent = await self._ensure_agent_started(AgentType.SMILES_ANALYSIS_AGENT)
            request = AgentRequest(instruction=query, parameters={"smiles": smiles})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            properties = output.get("properties", {})
        except Exception as e:
            logger.error(f"SMILES agent error: {e}")
            properties = {}
        
        formatted_smiles = self._format_smiles(smiles, properties)
        if formatted_smiles is None:
            formatted_smiles = f"## 🧪 SMILES Analysis\n\n**SMILES:** `{smiles}`\nNo properties available."
        
        llm_explanation = ""
        if properties:
            prompt = (
                f"Explain these molecular properties in plain language:\n\n"
                f"SMILES: {smiles}\n"
                f"Properties: {properties}\n\n"
                "Explain:\n"
                "1. What kind of molecule this is\n"
                "2. What these properties mean for its behavior\n"
                "3. Is it drug-like? (Lipinski's Rule of 5)\n"
                "4. Any interesting features\n\n"
                "Keep it simple and conversational."
            )
            
            llm_explanation = await self._call_llm(prompt)
            if llm_explanation:
                llm_explanation = f"\n\n🧪 **Chemistry Explanation:**\n{llm_explanation}"
            else:
                llm_explanation = ""
        
        return {"answer": formatted_smiles + llm_explanation, "properties": properties, "confidence": 0.9}

    async def _handle_knowledge_graph(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle knowledge graph queries."""
        entities = re.findall(r'\b([A-Z][A-Z0-9]{1,7})\b', query)
        if not entities:
            return {
                "answer": "No gene/protein found in your query. Please specify a gene or protein (e.g., EGFR, BRCA1).",
                "confidence": 0.5
            }
        
        try:
            agent = await self._ensure_agent_started(AgentType.KNOWLEDGE_GRAPH_AGENT)
            request = AgentRequest(instruction=query, parameters={"entity1": entities[0]})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            neighbors = output.get("neighbors", [])
        except Exception as e:
            logger.error(f"KG agent error: {e}")
            neighbors = []
        
        formatted_kg = self._format_knowledge_graph(entities[0], neighbors)
        if formatted_kg is None:
            formatted_kg = f"🕸️ No relationships found for {entities[0]}."
        
        if neighbors:
            prompt = (
                f"Explain these biological relationships for {entities[0]}:\n\n"
                f"{neighbors[:15]}\n\n"
                "Provide:\n"
                "1. What the entity is and its role\n"
                "2. What these connections mean\n"
                "3. Any significant pathways or interactions\n"
                "4. Why this matters for biomedical research\n\n"
                "Be informative but accessible."
            )
            
            explanation = await self._call_llm(prompt)
            if explanation:
                formatted_kg += f"\n\n🧬 **Biological Context:**\n{explanation}"
        
        return {"answer": formatted_kg, "neighbors": neighbors, "confidence": 0.85}

    async def _handle_drug_discovery(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle drug discovery queries."""
        pattern = r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}'
        smiles_match = re.search(pattern, query)
        
        if not smiles_match:
            return {
                "answer": "🔬 No SMILES string found. Please provide a SMILES string for analysis.",
                "confidence": 0.5
            }
        
        smiles = smiles_match.group(0)
        
        try:
            agent = await self._ensure_agent_started(AgentType.DRUG_REPURPOSING_AGENT)
            request = AgentRequest(instruction=query, parameters={"smiles": smiles})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            candidates = output.get("candidates", [])
        except Exception as e:
            logger.error(f"Drug discovery agent error: {e}")
            candidates = []
        
        formatted_dd = self._format_drug_discovery(candidates)
        if formatted_dd is None:
            formatted_dd = "🔬 No candidates found."
        
        if candidates:
            prompt = (
                f"Analyze these drug discovery results:\n\n"
                f"{candidates[:5]}\n\n"
                "Recommend:\n"
                "1. The best candidate and why\n"
                "2. Strengths and weaknesses of each\n"
                "3. Next steps for validation\n\n"
                "Be specific and practical."
            )
            
            recommendation = await self._call_llm(prompt)
            if recommendation:
                formatted_dd += f"\n\n💊 **AI Recommendation:**\n{recommendation}"
        
        return {"answer": formatted_dd, "candidates": candidates, "confidence": 0.85}

    async def _handle_compare(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle comparison queries."""
        papers = getattr(self, '_cached_papers', [])
        
        if not papers:
            return {
                "answer": "No papers to compare. Please search for papers first.",
                "confidence": 0.5
            }
        
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            idx1, idx2 = int(numbers[0]), int(numbers[1])
            if 1 <= idx1 <= len(papers) and 1 <= idx2 <= len(papers):
                paper1 = papers[idx1 - 1]
                paper2 = papers[idx2 - 1]
                
                prompt = (
                    f"Compare these two research papers:\n\n"
                    f"Paper 1:\n"
                    f"Title: {paper1.get('title', 'Unknown')}\n"
                    f"Abstract: {paper1.get('abstract', 'N/A')[:500]}\n\n"
                    f"Paper 2:\n"
                    f"Title: {paper2.get('title', 'Unknown')}\n"
                    f"Abstract: {paper2.get('abstract', 'N/A')[:500]}\n\n"
                    "Compare them on:\n"
                    "1. Research focus and question\n"
                    "2. Methods\n"
                    "3. Key findings\n"
                    "4. Contribution and significance\n\n"
                    "Be specific and clear."
                )
                
                comparison = await self._call_llm(prompt)
                if comparison:
                    return {"answer": f"## 🔄 Paper Comparison\n\n{comparison}", "confidence": 0.9}
        
        return {
            "answer": "Please specify which papers to compare (e.g., 'compare paper 3 and paper 7')",
            "confidence": 0.5
        }

    async def _handle_general(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle general questions."""
        prompt = (
            f"User question: {query}\n\n"
            "Provide a helpful, informative response about biomedical research or drug discovery.\n"
            "If you don't know, say so clearly.\n"
            "Be conversational and engaging."
        )
        
        answer = await self._call_llm(prompt)
        if answer is None:
            answer = "I'm not sure how to answer that. Can you rephrase?"
        
        return {"answer": answer, "confidence": 0.7}

    # ============================================================
    # FORMATTING METHODS
    # ============================================================

    def _papers_for_llm(self, papers: list) -> str:
        """Format papers for LLM prompt."""
        if not papers:
            return "No papers available."
        
        lines = []
        for i, paper in enumerate(papers[:10], 1):
            title = paper.get('title', 'Unknown')
            authors = ', '.join(paper.get('authors', ['Unknown'])[:3])
            year = paper.get('year', 'n.d.')
            journal = paper.get('journal', 'Unknown')
            abstract = paper.get('abstract', 'No abstract')[:300]
            lines.append(f"{i}. {title}\n   {authors} ({year}) - {journal}\n   Abstract: {abstract}...\n")
        
        result = "\n".join(lines)
        return result if result else "No papers available."

    def _format_papers(self, papers: list) -> str:
        """Format papers for display."""
        if not papers:
            return "📚 No papers found. Please try a different search query."
        
        lines = [f"## 📚 Papers Found ({len(papers)})\n"]
        for i, paper in enumerate(papers[:10], 1):
            title = paper.get("title", "Untitled")
            authors = paper.get("authors", [])
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            year = paper.get("year", "n.d.")
            journal = paper.get("journal", "Unknown Journal")
            lines.append(f"**{i}. {title}**")
            lines.append(f"   {author_str} ({year}) - {journal}\n")
        
        if len(papers) > 10:
            lines.append(f"*... and {len(papers) - 10} more papers*")
        
        result = "\n".join(lines)
        return result if result else "📚 No papers found."

    def _format_paper_details(self, paper: dict, idx: int) -> str:
        """Format a single paper with details."""
        if not paper:
            return f"## 📄 Paper #{idx}\n\nPaper details not available."
        
        lines = [
            f"## 📄 Paper #{idx}",
            "",
            f"**Title:** {paper.get('title', 'Untitled')}",
            f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}",
            f"**Year:** {paper.get('year', 'n.d.')}",
            f"**Journal:** {paper.get('journal', 'Unknown')}",
            f"**DOI:** {paper.get('doi', 'N/A')}",
            f"**PMID:** {paper.get('pmid', 'N/A')}",
            "",
            f"**Abstract:**\n{paper.get('abstract', 'No abstract available.')}",
        ]
        return "\n".join(lines)

    def _format_drug(self, drug_data: dict) -> str:
        """Format drug information."""
        if not drug_data:
            return "💊 Drug information not found."
        
        compound = drug_data.get("compound", {})
        if not compound:
            return "💊 Drug information not found."
        
        lines = [
            f"## 💊 {compound.get('drug_name', 'Unknown Drug')}",
            "",
            f"**Molecular Weight:** {compound.get('molecular_weight', 'N/A')}",
            f"**LogP:** {compound.get('logp', 'N/A')}",
            f"**SMILES:** `{compound.get('canonical_smiles', 'N/A')}`",
            f"**Lipinski Violations:** {compound.get('lipinski_violations', 0)}",
        ]
        return "\n".join(lines)

    def _format_smiles(self, smiles: str, properties: dict) -> str:
        """Format SMILES analysis."""
        if not properties:
            return f"## 🧪 SMILES Analysis\n\n**SMILES:** `{smiles}`\nNo properties available."
        
        lines = [
            f"## 🧪 SMILES Analysis",
            "",
            f"**SMILES:** `{smiles}`",
            f"**Molecular Weight:** {properties.get('Exact Molecular Weight', 'N/A')}",
            f"**LogP:** {properties.get('LogP', 'N/A')}",
            f"**TPSA:** {properties.get('TPSA', 'N/A')}",
            f"**H-Bond Donors:** {properties.get('H-Bond Donors', 'N/A')}",
            f"**H-Bond Acceptors:** {properties.get('H-Bond Acceptors', 'N/A')}",
        ]
        return "\n".join(lines)

    def _format_knowledge_graph(self, entity: str, neighbors: list) -> str:
        """Format knowledge graph results."""
        if not neighbors:
            return f"🕸️ No relationships found for {entity}."
        
        lines = [
            f"## 🕸️ Relationships for {entity}",
            "",
            f"Found {len(neighbors)} related entities:",
        ]
        for n in neighbors[:15]:
            lines.append(f"- {n}")
        
        return "\n".join(lines)

    def _format_drug_discovery(self, candidates: list) -> str:
        """Format drug discovery results."""
        if not candidates:
            return "🔬 No candidates found."
        
        lines = [
            f"## 🔬 Drug Discovery Results",
            "",
            f"Found {len(candidates)} candidates:",
        ]
        for c in candidates[:5]:
            name = c.get('name', 'Unknown')
            score = c.get('overall_score', 0)
            status = c.get('status', 'pending')
            lines.append(f"- **{name}** (Score: {score:.3f}, Status: {status})")
        
        return "\n".join(lines)