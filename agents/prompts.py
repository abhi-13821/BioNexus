"""
agents/prompts.py

Centralized prompt templates for the BioNexus Multi-Agent AI System.

This module contains all prompt templates used by the various agents,
the planner, coordinator, and conversation manager. Prompts are organized
by category and support versioning, dynamic placeholders, and helper
functions for rendering.

Design decisions:
    - All prompts are centralized in one module for easy maintenance
    - Prompts are versioned to track changes over time
    - Placeholders use {variable} syntax for consistency
    - Helper functions render prompts with type-safe parameters
    - No hardcoded biomedical data in prompts

Compatibility
-------------
Targets Python 3.11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ----------------------------------------------------------------------
# Prompt versioning
# ----------------------------------------------------------------------


class PromptVersion(str, Enum):
    """Version identifiers for prompts."""

    V1 = "v1.0"
    V2 = "v2.0"
    LATEST = "v1.0"


@dataclass
class PromptTemplate:
    """
    A prompt template with versioning and metadata.

    Attributes:
        name: Unique name of the prompt.
        version: Version identifier.
        template: The prompt template string.
        description: Description of what the prompt is for.
        placeholders: List of placeholder names used in the template.
    """

    name: str
    version: str
    template: str
    description: str = ""
    placeholders: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """
        Render the prompt template with the given variables.

        Args:
            **kwargs: Variables to substitute into the template.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If a required placeholder is missing.
        """
        # Check for missing placeholders
        missing = [p for p in self.placeholders if p not in kwargs]
        if missing:
            raise KeyError(f"Missing placeholders: {', '.join(missing)}")

        return self.template.format(**kwargs)

    def render_with_defaults(self, defaults: dict[str, Any], **kwargs: Any) -> str:
        """
        Render the prompt template with defaults and overrides.

        Args:
            defaults: Default values for placeholders.
            **kwargs: Override values for placeholders.

        Returns:
            The rendered prompt string.
        """
        merged = {**defaults, **kwargs}
        return self.render(**merged)


# ----------------------------------------------------------------------
# Prompt registry
# ----------------------------------------------------------------------


class PromptRegistry:
    """
    Registry for managing and retrieving prompt templates.

    Attributes:
        prompts: Dictionary of prompt templates by name.
    """

    def __init__(self) -> None:
        """Initialize the prompt registry."""
        self._prompts: dict[str, PromptTemplate] = {}
        self._versions: dict[str, dict[str, PromptTemplate]] = {}  # name -> version -> prompt

    def register(self, prompt: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            prompt: The PromptTemplate to register.
        """
        self._prompts[prompt.name] = prompt

        if prompt.name not in self._versions:
            self._versions[prompt.name] = {}
        self._versions[prompt.name][prompt.version] = prompt

    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name and optional version.

        Args:
            name: The prompt name.
            version: The version to retrieve. If None, returns the latest.

        Returns:
            The PromptTemplate, or None if not found.
        """
        if version is None:
            # Get the latest version
            if name not in self._versions:
                return None
            versions = self._versions[name]
            # Assume latest is the highest version string
            latest = max(versions.keys())
            return versions[latest]

        if name in self._versions and version in self._versions[name]:
            return self._versions[name][version]

        return None

    def get_all(self, name: str) -> list[PromptTemplate]:
        """
        Get all versions of a prompt.

        Args:
            name: The prompt name.

        Returns:
            A list of PromptTemplate instances.
        """
        if name not in self._versions:
            return []
        return list(self._versions[name].values())

    def list_prompts(self) -> list[str]:
        """
        List all registered prompt names.

        Returns:
            A list of prompt names.
        """
        return list(self._prompts.keys())

    def render(self, name: str, version: Optional[str] = None, **kwargs: Any) -> str:
        """
        Render a prompt by name.

        Args:
            name: The prompt name.
            version: The version to use.
            **kwargs: Variables to substitute.

        Returns:
            The rendered prompt string.

        Raises:
            KeyError: If the prompt is not found or placeholders are missing.
        """
        prompt = self.get(name, version)
        if prompt is None:
            raise KeyError(f"Prompt '{name}' not found")
        return prompt.render(**kwargs)


# ----------------------------------------------------------------------
# Prompt definitions
# ----------------------------------------------------------------------


def create_prompt_registry() -> PromptRegistry:
    """
    Create and populate the prompt registry with all prompts.

    Returns:
        A populated PromptRegistry instance.
    """
    registry = PromptRegistry()

    # ------------------------------------------------------------------
    # LiteratureAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="literature_search",
            version=PromptVersion.V1,
            template="""Search for biomedical literature on the following topic:

Topic: {query}

Please search across multiple biomedical databases including:
- PubMed
- Europe PMC
- CrossRef
- Semantic Scholar
- OpenAlex
- DOAJ
- bioRxiv
- medRxiv

Focus on finding recent, high-quality research papers. Prioritize:
1. Peer-reviewed articles
2. Recent publications (last 5 years)
3. High citation count papers
4. Open access papers when available

Filters to apply:
- Year range: {year_from} to {year_to}
- Article type: {article_type}
- Open access only: {open_access_only}

Return results ranked by {sort_by}.

{additional_instructions}""",
            description="Prompt for literature search",
            placeholders=["query", "year_from", "year_to", "article_type", "open_access_only", "sort_by", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="literature_summarize",
            version=PromptVersion.V1,
            template="""Summarize the following research papers:

Papers:
{papers}

Provide a comprehensive summary covering:
1. Main research themes
2. Key findings
3. Methodology approaches
4. Clinical or research significance
5. Limitations
6. Future research directions

{additional_instructions}""",
            description="Prompt for summarizing literature search results",
            placeholders=["papers", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="literature_review",
            version=PromptVersion.V1,
            template="""Generate a structured literature review on:

Topic: {topic}

Based on the following papers:
{papers}

The review should include:
1. Introduction (background and context)
2. Current Research (major findings)
3. Important Findings (key contributions)
4. Research Gaps (what's missing)
5. Future Scope (recommendations)

Format as a professional research review.

{additional_instructions}""",
            description="Prompt for generating a literature review",
            placeholders=["topic", "papers", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # DrugAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="drug_search",
            version=PromptVersion.V1,
            template="""Retrieve comprehensive drug information for:

Drug/Compound: {drug_name}
SMILES: {smiles}

Provide detailed information including:
1. Basic chemical properties
2. Pharmacological classification
3. Mechanism of action
4. Indications and uses
5. Contraindications
6. Side effects and adverse reactions
7. Drug interactions
8. Pharmacokinetics (absorption, distribution, metabolism, excretion)
9. Dosage information
10. Regulatory status

{additional_instructions}""",
            description="Prompt for drug information search",
            placeholders=["drug_name", "smiles", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="drug_comparison",
            version=PromptVersion.V1,
            template="""Compare the following drugs:

Drug A: {drug_a_name}
Drug B: {drug_b_name}

Compare based on:
1. Chemical structure and properties
2. Mechanism of action
3. Therapeutic indications
4. Safety profile
5. Pharmacokinetics
6. Clinical efficacy
7. Contraindications
8. Drug interactions
9. Cost and availability

{additional_instructions}""",
            description="Prompt for comparing two drugs",
            placeholders=["drug_a_name", "drug_b_name", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="drug_mechanism",
            version=PromptVersion.V1,
            template="""Explain the mechanism of action for:

Drug: {drug_name}

Please provide a detailed, mechanistic explanation including:
1. Primary molecular target(s)
2. Binding interactions
3. Downstream signaling pathways
4. Physiological effects
5. Therapeutic implications
6. Potential off-target effects

{additional_instructions}""",
            description="Prompt for explaining drug mechanism of action",
            placeholders=["drug_name", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # SmilesAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="smiles_analyze",
            version=PromptVersion.V1,
            template="""Analyze the following chemical structure:

SMILES: {smiles}
Compound Name: {compound_name}

Provide a comprehensive structural analysis including:
1. Molecular formula
2. Molecular weight
3. Key functional groups
4. Lipinski Rule of Five assessment
5. Drug-likeness score
6. Predicted physicochemical properties
7. Structural alerts (if any)
8. 3D structure characteristics

{additional_instructions}""",
            description="Prompt for SMILES analysis",
            placeholders=["smiles", "compound_name", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="smiles_properties",
            version=PromptVersion.V1,
            template="""Explain the molecular properties of this compound:

SMILES: {smiles}

Please explain the following properties in plain language:
{molecular_properties}

Focus on:
1. What each property means for drug development
2. Whether the properties are favorable for drug-likeness
3. How the properties compare to typical drugs
4. Any potential issues or concerns

{additional_instructions}""",
            description="Prompt for explaining molecular properties",
            placeholders=["smiles", "molecular_properties", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="smiles_functional_groups",
            version=PromptVersion.V1,
            template="""Explain the functional groups present in this molecule:

SMILES: {smiles}
Functional Groups Detected: {functional_groups}

For each functional group, explain:
1. What it is and its chemical properties
2. Its role in drug-target interactions
3. Any implications for drug metabolism
4. Potential effects on pharmacokinetics

{additional_instructions}""",
            description="Prompt for explaining functional groups",
            placeholders=["smiles", "functional_groups", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # KnowledgeGraphAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="kg_query",
            version=PromptVersion.V1,
            template="""Explore relationships in the biomedical knowledge graph:

Entities: {entity1} and {entity2}

Find and explain connections between these entities including:
1. Direct relationships
2. Indirect paths through intermediate entities
3. Shared neighbors
4. Biological pathways involved
5. Clinical significance of the relationships
6. Potential mechanisms of interaction

{additional_instructions}""",
            description="Prompt for knowledge graph query",
            placeholders=["entity1", "entity2", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="kg_neighbors",
            version=PromptVersion.V1,
            template="""Find connected entities for:

Entity: {entity}

Discover related entities including:
1. Direct connections
2. Types of relationships
3. Strength of associations
4. Biological significance
5. Clinical implications
6. Potential research directions

{additional_instructions}""",
            description="Prompt for finding neighbors in knowledge graph",
            placeholders=["entity", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="kg_pathway",
            version=PromptVersion.V1,
            template="""Explain biological pathways involving:

Entities: {entities}

Describe the pathways including:
1. Pathway components
2. Biological processes
3. Regulatory mechanisms
4. Disease associations
5. Therapeutic implications
6. Research opportunities

{additional_instructions}""",
            description="Prompt for pathway exploration",
            placeholders=["entities", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # DrugDiscoveryAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="discovery_analysis",
            version=PromptVersion.V1,
            template="""Analyze the following compound for drug discovery:

Compound Name: {compound_name}
SMILES: {smiles}

Provide a comprehensive drug discovery analysis including:
1. Molecular properties and drug-likeness
2. Potential biological targets
3. Toxicity predictions
4. Similar compounds and SAR insights
5. Repurposing opportunities
6. Development challenges
7. Research and development recommendations

{additional_instructions}""",
            description="Prompt for drug discovery analysis",
            placeholders=["compound_name", "smiles", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="discovery_ranking",
            version=PromptVersion.V1,
            template="""Rank and analyze drug discovery candidates:

Candidates: {candidates}

Provide a detailed ranking analysis including:
1. Overall ranking and justification
2. Strengths and weaknesses of each candidate
3. Comparison across key criteria
4. Risk assessment
5. Recommendation for prioritization
6. Next steps for development

{additional_instructions}""",
            description="Prompt for ranking drug discovery candidates",
            placeholders=["candidates", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # PlannerAgent prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="plan_intent",
            version=PromptVersion.V1,
            template="""Analyze the user's query and determine the intent:

User Query: {query}

Available agents:
{available_agents}

Based on the query, determine:
1. The primary intent
2. Secondary intents (if any)
3. Which agents should be invoked
4. The execution order
5. Any dependencies between agents
6. Whether clarification is needed
7. Confidence level for the plan

{additional_instructions}""",
            description="Prompt for intent detection and planning",
            placeholders=["query", "available_agents", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="plan_clarification",
            version=PromptVersion.V1,
            template="""Generate clarification questions for an ambiguous query:

User Query: {query}
Detected Intents: {intents}
Missing Information: {missing_info}

Generate clear, concise questions to clarify the user's intent.

{additional_instructions}""",
            description="Prompt for generating clarification questions",
            placeholders=["query", "intents", "missing_info", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # Coordinator prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="coordinator_merge",
            version=PromptVersion.V1,
            template="""Merge and synthesize results from multiple agents:

Query: {query}
Agent Results:
{agent_results}

Synthesize a unified response that:
1. Integrates findings from all agents
2. Highlights key insights
3. Resolves any conflicting information
4. Provides a coherent narrative
5. Answers the original query completely

{additional_instructions}""",
            description="Prompt for merging agent results",
            placeholders=["query", "agent_results", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="coordinator_fallback",
            version=PromptVersion.V1,
            template="""Provide a helpful response when an agent fails:

Query: {query}
Failed Agent: {failed_agent}
Error: {error}

Generate a response that:
1. Acknowledges the issue
2. Provides whatever information is available
3. Explains what went wrong
4. Suggests next steps or alternatives
5. Offers to retry or help with something else

{additional_instructions}""",
            description="Prompt for fallback responses",
            placeholders=["query", "failed_agent", "error", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # ConversationManager prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="conversation_welcome",
            version=PromptVersion.V1,
            template="""Welcome to BioNexus AI Assistant.

I can help you with:
- Literature Search and analysis
- Drug Information and pharmacology
- Chemical Structure analysis (SMILES)
- Knowledge Graph exploration
- Drug Discovery and candidate evaluation

How can I assist you today?

{additional_instructions}""",
            description="Prompt for conversation welcome message",
            placeholders=["additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="conversation_followup",
            version=PromptVersion.V1,
            template="""Continue the conversation about:

Previous Context:
{context}

Previous User Query: {previous_query}
Previous Assistant Response: {previous_response}

New User Query: {new_query}

Provide a response that:
1. Acknowledges the conversation history
2. Builds on previous information
3. Answers the new query in context
4. Maintains consistency with earlier responses

{additional_instructions}""",
            description="Prompt for follow-up conversations",
            placeholders=["context", "previous_query", "previous_response", "new_query", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="conversation_summary",
            version=PromptVersion.V1,
            template="""Summarize the conversation history:

Conversation Messages:
{messages}

Generate a concise summary including:
1. Main topics discussed
2. Key information discovered
3. Important conclusions
4. Pending questions or topics

{additional_instructions}""",
            description="Prompt for summarizing conversation history",
            placeholders=["messages", "additional_instructions"],
        )
    )

    # ------------------------------------------------------------------
    # General Q&A prompts
    # ------------------------------------------------------------------

    registry.register(
        PromptTemplate(
            name="general_qa",
            version=PromptVersion.V1,
            template="""Answer the following biomedical research question:

Question: {query}

Provide a clear, evidence-based answer that:
1. Directly addresses the question
2. Provides relevant context
3. Cites supporting information
4. Acknowledges limitations or uncertainties
5. Suggests further reading if appropriate

{additional_instructions}""",
            description="Prompt for general Q&A",
            placeholders=["query", "additional_instructions"],
        )
    )

    registry.register(
        PromptTemplate(
            name="error_handling",
            version=PromptVersion.V1,
            template="""A user encountered an error:

Error Message: {error}
Context: {context}

Generate a helpful, empathetic response that:
1. Acknowledges the issue
2. Explains what happened (without technical jargon)
3. Suggests a solution or workaround
4. Offers alternative ways to help
5. Maintains a positive and supportive tone

{additional_instructions}""",
            description="Prompt for error handling",
            placeholders=["error", "context", "additional_instructions"],
        )
    )

    return registry


# ----------------------------------------------------------------------
# Helper functions for rendering prompts
# ----------------------------------------------------------------------


def get_prompt_registry() -> PromptRegistry:
    """
    Get the singleton prompt registry.

    Returns:
        The PromptRegistry instance.
    """
    return create_prompt_registry()


def render_prompt(prompt_name: str, **kwargs: Any) -> str:
    """
    Render a prompt by name with the given variables.

    Args:
        prompt_name: The name of the prompt to render.
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.

    Raises:
        KeyError: If the prompt is not found or placeholders are missing.
    """
    registry = get_prompt_registry()
    return registry.render(prompt_name, **kwargs)


def render_prompt_with_defaults(prompt_name: str, defaults: dict[str, Any], **kwargs: Any) -> str:
    """
    Render a prompt with defaults and overrides.

    Args:
        prompt_name: The name of the prompt to render.
        defaults: Default values for placeholders.
        **kwargs: Override values for placeholders.

    Returns:
        The rendered prompt string.

    Raises:
        KeyError: If the prompt is not found.
    """
    registry = get_prompt_registry()
    prompt = registry.get(prompt_name)
    if prompt is None:
        raise KeyError(f"Prompt '{prompt_name}' not found")
    return prompt.render_with_defaults(defaults, **kwargs)


def render_literature_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Render a literature-related prompt.

    Args:
        prompt_type: The type of literature prompt ('search', 'summarize', 'review').
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.
    """
    prompt_names = {
        "search": "literature_search",
        "summarize": "literature_summarize",
        "review": "literature_review",
    }
    prompt_name = prompt_names.get(prompt_type, "literature_search")
    return render_prompt(prompt_name, **kwargs)


def render_drug_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Render a drug-related prompt.

    Args:
        prompt_type: The type of drug prompt ('search', 'comparison', 'mechanism').
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.
    """
    prompt_names = {
        "search": "drug_search",
        "comparison": "drug_comparison",
        "mechanism": "drug_mechanism",
    }
    prompt_name = prompt_names.get(prompt_type, "drug_search")
    return render_prompt(prompt_name, **kwargs)


def render_smiles_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Render a SMILES-related prompt.

    Args:
        prompt_type: The type of SMILES prompt ('analyze', 'properties', 'functional_groups').
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.
    """
    prompt_names = {
        "analyze": "smiles_analyze",
        "properties": "smiles_properties",
        "functional_groups": "smiles_functional_groups",
    }
    prompt_name = prompt_names.get(prompt_type, "smiles_analyze")
    return render_prompt(prompt_name, **kwargs)


def render_kg_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Render a knowledge graph-related prompt.

    Args:
        prompt_type: The type of KG prompt ('query', 'neighbors', 'pathway').
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.
    """
    prompt_names = {
        "query": "kg_query",
        "neighbors": "kg_neighbors",
        "pathway": "kg_pathway",
    }
    prompt_name = prompt_names.get(prompt_type, "kg_query")
    return render_prompt(prompt_name, **kwargs)


def render_discovery_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Render a drug discovery-related prompt.

    Args:
        prompt_type: The type of discovery prompt ('analysis', 'ranking').
        **kwargs: Variables to substitute.

    Returns:
        The rendered prompt string.
    """
    prompt_names = {
        "analysis": "discovery_analysis",
        "ranking": "discovery_ranking",
    }
    prompt_name = prompt_names.get(prompt_type, "discovery_analysis")
    return render_prompt(prompt_name, **kwargs)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

__all__: list[str] = [
    "PromptVersion",
    "PromptTemplate",
    "PromptRegistry",
    "create_prompt_registry",
    "get_prompt_registry",
    "render_prompt",
    "render_prompt_with_defaults",
    "render_literature_prompt",
    "render_drug_prompt",
    "render_smiles_prompt",
    "render_kg_prompt",
    "render_discovery_prompt",
]