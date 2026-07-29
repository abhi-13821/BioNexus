"""
agents/smart_coordinator.py

Smart Coordinator with full LLM integration for conversational agents.
Uses Ollama (phi3:mini) for intelligent responses and explanations.
"""

from __future__ import annotations

import json
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
        model: str = "phi3:mini",
    ):
        super().__init__(config=config, registry=registry, memory=memory)
        self.ollama_url = ollama_url
        self.model = model
        self._ollama_available = self._check_ollama()
        self._initialized = False
        
        if self._ollama_available:
            logger.info(f"✅ Ollama available with model: {model}")
        else:
            logger.warning("⚠️ Ollama not available. Using fallback mode.")

    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": "test", "stream": False},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    async def initialize(self) -> None:
        """Initialize the coordinator."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("✅ SmartCoordinator initialized")

    async def process_query(self, query: str, conversation_id: Optional[str] = None, **kwargs) -> CoordinatorResponse:
        """Process a user query with LLM intelligence."""

        # Step 1: Detect intent
        intent = self._detect_intent(query)
        
        # Step 2: Get context from memory
        context = self._get_context(conversation_id)
        
        # Step 3: Route to appropriate handler
        if intent["type"] == "literature":
            result = await self._handle_literature(query, context, conversation_id)
        elif intent["type"] == "drug":
            result = await self._handle_drug(query, context, conversation_id)
        elif intent["type"] == "smiles":
            result = await self._handle_smiles(query, context, conversation_id)
        elif intent["type"] == "knowledge_graph":
            result = await self._handle_knowledge_graph(query, context, conversation_id)
        elif intent["type"] == "drug_discovery":
            result = await self._handle_drug_discovery(query, context, conversation_id)
        elif intent["type"] == "follow_up":
            result = await self._handle_follow_up(query, context, conversation_id)
        elif intent["type"] == "compare":
            result = await self._handle_compare(query, context, conversation_id)
        else:
            result = await self._handle_general(query, context, conversation_id)
        
        # Step 4: Store in memory
        if conversation_id:
            try:
                self.memory.remember_turn(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=result["answer"],
                    agent_type=AgentType.ORCHESTRATOR,
                )
            except Exception:
                pass
        
        # Step 5: Return response
        return CoordinatorResponse(
            request_id=generate_id("SMART"),
            query=query,
            status=TaskStatus.COMPLETED,
            message=result["answer"],
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
        
        # Follow-up detection for papers
        if re.search(r'paper\s*#?\s*(\d+)|#\s*(\d+)', q):
            return {"type": "follow_up", "subtype": "paper_detail"}
        if "summarize" in q or "summary" in q or "overview" in q:
            return {"type": "follow_up", "subtype": "summarize"}
        if "compare" in q:
            return {"type": "compare"}
        
        # Literature search
        if any(w in q for w in ["paper", "papers", "publication", "research", "study", "literature", "find"]):
            return {"type": "literature", "query": query}
        
        # Drug info
        if any(w in q for w in ["drug", "medicine", "medication", "compound", "pharma"]):
            return {"type": "drug", "query": query}
        
        # SMILES analysis
        if any(w in q for w in ["smiles", "structure", "molecule", "analysis"]):
            return {"type": "smiles", "query": query}
        
        # Knowledge graph
        if any(w in q for w in ["relationship", "connection", "pathway", "gene", "protein"]):
            return {"type": "knowledge_graph", "query": query}
        
        # Drug discovery
        if any(w in q for w in ["discovery", "candidate", "similar", "repurpose"]):
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
            logger.warning("Ollama not available")
            return None
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json().get("response")
                return result.strip()
            else:
                logger.error(f"Ollama error: {response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
        return None

    async def _handle_literature(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle literature search with LLM explanation."""
        # Get papers from agent
        try:
            agent = self.registry.get_instance(AgentType.LITERATURE_SEARCH_AGENT)
            request = AgentRequest(instruction=query, parameters={"query": query})
            response = await agent.handle_request(request)
            
            papers = []
            if response.results and response.results[0].output:
                papers = response.results[0].output.get("papers", [])
        except Exception as e:
            logger.error(f"Literature agent error: {e}")
            papers = []
        
        # Format papers
        formatted_papers = self._format_papers(papers)
        
        # Generate LLM summary if papers exist
        llm_summary = ""
        if papers and len(papers) > 0:
            prompt = f"""Summarize these {len(papers)} research papers about {query[:50]}...:
            
{self._papers_for_llm(papers[:5])}

Provide:
1. A brief overview (2-3 sentences)
2. Key themes and findings
3. Most interesting/important paper and why
4. A simple recommendation for further reading

Keep it concise and conversational."""
            
            llm_summary = await self._call_llm(prompt)
            if llm_summary:
                llm_summary = f"\n\n🤖 **AI Summary:**\n{llm_summary}"
        
        # Cache papers for follow-up
        self._cached_papers = papers
        self._cached_query = query
        
        answer = formatted_papers + llm_summary
        return {"answer": answer, "papers": papers, "count": len(papers), "confidence": 0.9}

    async def _handle_drug(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle drug information with LLM explanation."""
        try:
            agent = self.registry.get_instance(AgentType.DRUG_INFO_AGENT)
            request = AgentRequest(instruction=query, parameters={"query": query})
            response = await agent.handle_request(request)
            
            drug_data = {}
            if response.results and response.results[0].output:
                drug_data = response.results[0].output
        except Exception as e:
            logger.error(f"Drug agent error: {e}")
            drug_data = {}
        
        formatted_drug = self._format_drug(drug_data)
        
        # LLM explanation
        llm_explanation = ""
        if drug_data:
            prompt = f"""Explain this drug information in simple, clear terms:
            
{drug_data}

Focus on:
1. What the drug is and what it's used for
2. How it works (mechanism of action)
3. Important properties
4. Anything interesting or unique about it

Be conversational and helpful."""
            
            llm_explanation = await self._call_llm(prompt)
            if llm_explanation:
                llm_explanation = f"\n\n🤖 **AI Explanation:**\n{llm_explanation}"
        
        return {"answer": formatted_drug + llm_explanation, "drug_data": drug_data, "confidence": 0.9}

    async def _handle_smiles(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle SMILES analysis with LLM explanation."""
        import re
        smiles_match = re.search(r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}', query)
        if not smiles_match:
            return {"answer": "🔬 No SMILES string found in your query. Please provide a SMILES string.", "confidence": 0.5}
        
        smiles = smiles_match.group(0)
        
        try:
            agent = self.registry.get_instance(AgentType.SMILES_ANALYSIS_AGENT)
            request = AgentRequest(instruction=query, parameters={"smiles": smiles})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            properties = output.get("properties", {})
        except Exception as e:
            logger.error(f"SMILES agent error: {e}")
            properties = {}
        
        formatted_smiles = self._format_smiles(smiles, properties)
        
        # LLM explanation
        llm_explanation = ""
        if properties:
            prompt = f"""Explain these molecular properties in plain language:
            
SMILES: {smiles}
Properties: {properties}

Explain:
1. What kind of molecule this is
2. What these properties mean for its behavior
3. Is it drug-like? (Lipinski's Rule of 5)
4. Any interesting features

Keep it simple and conversational."""
            
            llm_explanation = await self._call_llm(prompt)
            if llm_explanation:
                llm_explanation = f"\n\n🧪 **Chemistry Explanation:**\n{llm_explanation}"
        
        return {"answer": formatted_smiles + llm_explanation, "properties": properties, "confidence": 0.9}

    async def _handle_follow_up(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle follow-up questions about previous results."""
        # Get cached papers from the last search
        papers = getattr(self, '_cached_papers', [])
        
        if not papers and not context:
            return {"answer": "I don't have previous results to refer to. Please do a search first.", "confidence": 0.5}
        
        # Check if asking about a specific paper
        match = re.search(r'paper\s*#?\s*(\d+)|#\s*(\d+)', query.lower())
        if match and papers:
            idx = int(match.group(1) or match.group(2))
            if 1 <= idx <= len(papers):
                paper = papers[idx - 1]
                
                # Use LLM to explain this paper
                prompt = f"""Explain this research paper in simple, clear terms:

Title: {paper.get('title', 'Unknown Title')}
Authors: {', '.join(paper.get('authors', ['Unknown']))}
Year: {paper.get('year', 'n.d.')}
Journal: {paper.get('journal', 'Unknown')}
Abstract: {paper.get('abstract', 'No abstract available.')[:1500]}

Provide:
1. What was the main question or problem?
2. What did they do (methods)?
3. What did they find (key results)?
4. Why does this matter (significance)?

Make it easy to understand for a non-expert. Be conversational."""
                
                explanation = await self._call_llm(prompt)
                if explanation:
                    return {
                        "answer": f"## 📄 Paper #{idx}\n\n{explanation}",
                        "paper": paper,
                        "confidence": 0.95
                    }
                else:
                    # Fallback to basic info
                    return {
                        "answer": self._format_paper_details(paper, idx),
                        "paper": paper,
                        "confidence": 0.8
                    }
        
        # Check if asking for summary
        if "summarize" in query.lower() or "summary" in query.lower():
            if papers:
                prompt = f"""Summarize these {len(papers)} research papers:
                
{self._papers_for_llm(papers[:10])}

Provide:
1. Overall summary of the research
2. Key themes that appear
3. Most significant findings
4. Gaps or future directions

Keep it concise and useful."""
                
                summary = await self._call_llm(prompt)
                if summary:
                    return {
                        "answer": f"## 📚 Research Summary\n\n{summary}",
                        "confidence": 0.9
                    }
        
        # Generic follow-up with context
        if context:
            prompt = f"""Previous conversation context:
{context}

User's follow-up question: {query}

Answer the question based on the context. If you don't know, say so clearly."""
            
            answer = await self._call_llm(prompt)
            if answer:
                return {"answer": answer, "confidence": 0.85}
        
        return {"answer": "I'm not sure how to answer that. Could you be more specific?", "confidence": 0.5}

    async def _handle_compare(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle comparison queries."""
        papers = getattr(self, '_cached_papers', [])
        if not papers:
            return {"answer": "No papers to compare. Please search for papers first.", "confidence": 0.5}
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            idx1, idx2 = int(numbers[0]), int(numbers[1])
            if 1 <= idx1 <= len(papers) and 1 <= idx2 <= len(papers):
                paper1 = papers[idx1 - 1]
                paper2 = papers[idx2 - 1]
                
                prompt = f"""Compare these two research papers:

Paper 1:
Title: {paper1.get('title', 'Unknown')}
Abstract: {paper1.get('abstract', 'N/A')[:500]}

Paper 2:
Title: {paper2.get('title', 'Unknown')}
Abstract: {paper2.get('abstract', 'N/A')[:500]}

Compare them on:
1. Research focus and question
2. Methods
3. Key findings
4. Contribution and significance

Be specific and clear."""
                
                comparison = await self._call_llm(prompt)
                if comparison:
                    return {
                        "answer": f"## 🔄 Paper Comparison\n\n{comparison}",
                        "confidence": 0.9
                    }
        
        return {"answer": "Please specify which papers to compare (e.g., 'compare paper 3 and paper 7')", "confidence": 0.5}

    async def _handle_general(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle general questions."""
        prompt = f"""User question: {query}

Provide a helpful, informative response about biomedical research or drug discovery.
If you don't know, say so clearly.
Be conversational and engaging."""
        
        answer = await self._call_llm(prompt)
        return {
            "answer": answer or "I'm not sure how to answer that. Can you rephrase?",
            "confidence": 0.7
        }

    async def _handle_knowledge_graph(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle knowledge graph queries."""
        import re
        entities = re.findall(r'\b([A-Z][A-Z0-9]{1,7})\b', query)
        if not entities:
            return {"answer": "No gene/protein found in your query.", "confidence": 0.5}
        
        try:
            agent = self.registry.get_instance(AgentType.KNOWLEDGE_GRAPH_AGENT)
            request = AgentRequest(instruction=query, parameters={"entity1": entities[0]})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            neighbors = output.get("neighbors", [])
        except Exception as e:
            logger.error(f"KG agent error: {e}")
            neighbors = []
        
        formatted_kg = self._format_knowledge_graph(entities[0], neighbors)
        
        if neighbors:
            prompt = f"""Explain these biological relationships for {entities[0]}:

{neighbors[:15]}

Provide:
1. What {entities[0]} is and its role
2. What these connections mean
3. Any significant pathways or interactions
4. Why this matters for biomedical research

Be informative but accessible."""
            
            explanation = await self._call_llm(prompt)
            if explanation:
                formatted_kg += f"\n\n🧬 **Biological Context:**\n{explanation}"
        
        return {"answer": formatted_kg, "neighbors": neighbors, "confidence": 0.85}

    async def _handle_drug_discovery(self, query: str, context: str, conversation_id: str) -> dict:
        """Handle drug discovery queries."""
        import re
        smiles_match = re.search(r'[A-Za-z0-9@+\-\[\]\(\)=#:$/\\%.~]{10,}', query)
        if not smiles_match:
            return {"answer": "🔬 No SMILES string found. Please provide a SMILES string for analysis.", "confidence": 0.5}
        
        smiles = smiles_match.group(0)
        
        try:
            agent = self.registry.get_instance(AgentType.DRUG_REPURPOSING_AGENT)
            request = AgentRequest(instruction=query, parameters={"smiles": smiles})
            response = await agent.handle_request(request)
            
            output = response.results[0].output if response.results else {}
            candidates = output.get("candidates", [])
        except Exception as e:
            logger.error(f"Drug discovery agent error: {e}")
            candidates = []
        
        formatted_dd = self._format_drug_discovery(candidates)
        
        if candidates:
            prompt = f"""Analyze these drug discovery results:

{candidates[:5]}

Recommend:
1. The best candidate and why
2. Strengths and weaknesses of each
3. Next steps for validation

Be specific and practical."""
            
            recommendation = await self._call_llm(prompt)
            if recommendation:
                formatted_dd += f"\n\n💊 **AI Recommendation:**\n{recommendation}"
        
        return {"answer": formatted_dd, "candidates": candidates, "confidence": 0.85}

    # ----------------------------------------------------------------------
    # Formatting Methods
    # ----------------------------------------------------------------------

    def _papers_for_llm(self, papers: list) -> str:
        """Format papers for LLM prompt."""
        lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Unknown')
            authors = ', '.join(paper.get('authors', ['Unknown'])[:3])
            year = paper.get('year', 'n.d.')
            journal = paper.get('journal', 'Unknown')
            abstract = paper.get('abstract', 'No abstract')[:300]
            lines.append(f"{i}. {title}\n   {authors} ({year}) - {journal}\n   Abstract: {abstract}...\n")
        return "\n".join(lines)

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
        
        return "\n".join(lines)

    def _format_paper_details(self, paper: dict, idx: int) -> str:
        """Format a single paper with details."""
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