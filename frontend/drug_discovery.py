"""
frontend/drug_discovery.py

Streamlit frontend for the BioNexus Drug Discovery module.

This module provides a modern, responsive web interface for the Drug Discovery
pipeline. It accepts compound inputs, executes the pipeline via the backend,
and visualizes the results in a clean, organized layout with appropriate
progress indicators, expandable sections, and export capabilities.

Integration Points:
    - drug_discovery.pipeline: run_drug_discovery, DrugDiscoveryPipeline
    - drug_discovery.models: All data models for display
    - drug_discovery.utils: Formatting and utility functions

Compatibility
-------------
Targets Python 3.11 with Streamlit.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from io import StringIO
from typing import Optional

import pandas as pd
import streamlit as st

from drug_discovery.models import (
    CandidateRanking,
    Compound,
    DiscoveryStage,
    DrugCandidate,
    DrugDiscoveryResult,
    MolecularProperties,
    PredictionConfidence,
    TargetPrediction,
    ToxicityLevel,
    ToxicityPrediction,
)
from drug_discovery.pipeline import (
    PipelineConfig,
    run_drug_discovery,
)
from drug_discovery.utils import (
    format_iso_timestamp,
    format_molecular_weight,
    format_percentage,
    format_similarity_score,
)

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="BioNexus Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------

st.markdown("""
<style>
    .main {
        padding: 0 1rem;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-container h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-container p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        padding: 0.5rem 0;
        border-bottom: 3px solid #4299e1;
        margin-bottom: 1rem;
    }
    
    .badge-very_high {
        background: #48bb78;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-high {
        background: #68d391;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-medium {
        background: #ecc94b;
        color: #1a202c;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-low {
        background: #fc8181;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .toxicity-low {
        color: #48bb78;
        font-weight: 600;
    }
    
    .toxicity-moderate {
        color: #ecc94b;
        font-weight: 600;
    }
    
    .toxicity-high {
        color: #ed8936;
        font-weight: 600;
    }
    
    .toxicity-severe {
        color: #fc8181;
        font-weight: 600;
    }
    
    .toxicity-unknown {
        color: #a0aec0;
        font-weight: 600;
    }
    
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #4299e1, #48bb78);
    }
    
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "result": None,
        "is_running": False,
        "smiles_input": "",
        "compound_name": "",
        "disease_name": "",
        "query_smiles": "",
        "show_advanced": False,
        "top_k": 25,
        "similarity_threshold": 0.7,
        "enable_cache": True,
        "enable_similarity": True,
        "enable_target_prediction": True,
        "enable_toxicity_prediction": True,
        "enable_repurposing": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# ----------------------------------------------------------------------
# Utility functions for display
# ----------------------------------------------------------------------

def get_confidence_badge(confidence: PredictionConfidence) -> str:
    """Return HTML badge for a confidence level."""
    badge_classes = {
        PredictionConfidence.VERY_HIGH: "badge-very_high",
        PredictionConfidence.HIGH: "badge-high",
        PredictionConfidence.MEDIUM: "badge-medium",
        PredictionConfidence.LOW: "badge-low",
    }
    return f'<span class="{badge_classes.get(confidence, "badge-low")}">{confidence.value.upper()}</span>'


def get_toxicity_class(toxicity_level: ToxicityLevel) -> str:
    """Return CSS class for a toxicity level."""
    classes = {
        ToxicityLevel.LOW: "toxicity-low",
        ToxicityLevel.MODERATE: "toxicity-moderate",
        ToxicityLevel.HIGH: "toxicity-high",
        ToxicityLevel.SEVERE: "toxicity-severe",
        ToxicityLevel.UNKNOWN: "toxicity-unknown",
    }
    return classes.get(toxicity_level, "toxicity-unknown")


def format_confidence(score: float) -> str:
    """Format a confidence score as a percentage."""
    return format_percentage(score, precision=1)


def format_molecular_weight_display(mw: float) -> str:
    """Format molecular weight for display."""
    return format_molecular_weight(mw, precision=2)


def format_similarity_display(score: float, metric: Optional[str] = None) -> str:
    """Format similarity score for display."""
    return format_similarity_score(score, metric, precision=3)


def create_compound_from_input(smiles: str, name: str) -> Compound:
    """Create a Compound instance from input values."""
    compound_id = f"INPUT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return Compound(
        compound_id=compound_id,
        name=name or "Input Compound",
        smiles=smiles.strip(),
    )


# ----------------------------------------------------------------------
# Input section
# ----------------------------------------------------------------------

def render_input_section() -> tuple[Optional[list[Compound]], Optional[str], Optional[str]]:
    """
    Render the compound input section.

    Returns:
        Tuple of (compounds, query_smiles, disease_name) or (None, None, None).
    """
    with st.container():
        st.markdown("### 🔬 Compound Input")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**SMILES String**")
            smiles_input = st.text_area(
                "Enter SMILES string",
                value=st.session_state.smiles_input,
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                height=80,
                key="smiles_area",
                label_visibility="collapsed",
            )
            st.session_state.smiles_input = smiles_input

            st.markdown("**Compound Name (optional)**")
            compound_name = st.text_input(
                "Compound name",
                value=st.session_state.compound_name,
                placeholder="e.g., Aspirin",
                key="name_input",
                label_visibility="collapsed",
            )
            st.session_state.compound_name = compound_name

        with col2:
            st.markdown("**Query SMILES (optional)**")
            query_smiles = st.text_input(
                "Query SMILES for similarity search",
                value=st.session_state.query_smiles,
                placeholder="Leave empty to use the input SMILES",
                key="query_smiles_input",
                label_visibility="collapsed",
            )
            st.session_state.query_smiles = query_smiles

            st.markdown("**Disease Name (optional)**")
            disease_name = st.text_input(
                "Disease for repurposing",
                value=st.session_state.disease_name,
                placeholder="e.g., type 2 diabetes",
                key="disease_input",
                label_visibility="collapsed",
            )
            st.session_state.disease_name = disease_name

        # Advanced configuration expander
        with st.expander("⚙️ Advanced Configuration", expanded=st.session_state.show_advanced):
            st.session_state.show_advanced = True

            col3, col4, col5 = st.columns(3)

            with col3:
                st.session_state.top_k = st.number_input(
                    "Top K candidates",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.top_k,
                    help="Maximum number of candidates to return",
                )

                st.session_state.similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.similarity_threshold,
                    step=0.05,
                    help="Minimum similarity score for candidates",
                )

            with col4:
                st.session_state.enable_cache = st.checkbox(
                    "Enable Caching",
                    value=st.session_state.enable_cache,
                    help="Use cache to avoid recomputing results",
                )

                st.session_state.enable_similarity = st.checkbox(
                    "Enable Similarity Search",
                    value=st.session_state.enable_similarity,
                    help="Perform compound similarity search",
                )

            with col5:
                st.session_state.enable_target_prediction = st.checkbox(
                    "Enable Target Prediction",
                    value=st.session_state.enable_target_prediction,
                    help="Predict biological targets",
                )

                st.session_state.enable_toxicity_prediction = st.checkbox(
                    "Enable Toxicity Prediction",
                    value=st.session_state.enable_toxicity_prediction,
                    help="Predict toxicity risk",
                )

                st.session_state.enable_repurposing = st.checkbox(
                    "Enable Drug Repurposing",
                    value=st.session_state.enable_repurposing,
                    help="Perform drug repurposing analysis",
                )

        # Run button
        col_run, col_reset = st.columns([3, 1])

        with col_run:
            run_button = st.button(
                "🚀 Run Drug Discovery Analysis",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_running or not smiles_input.strip(),
            )

        with col_reset:
            reset_button = st.button(
                "🔄 Reset",
                use_container_width=True,
                disabled=st.session_state.is_running,
            )

        if reset_button:
            st.session_state.result = None
            st.session_state.is_running = False
            st.rerun()

        # Input validation
        if run_button:
            if not smiles_input.strip():
                st.error("Please enter a valid SMILES string.")
                return None, None, None

            return [create_compound_from_input(smiles_input, compound_name)], query_smiles or smiles_input, disease_name

        return None, None, None


# ----------------------------------------------------------------------
# Results rendering functions
# ----------------------------------------------------------------------

def render_execution_stats(result: DrugDiscoveryResult) -> None:
    """Render execution statistics."""
    st.markdown("### 📊 Execution Statistics")

    metadata = result.metadata

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Run ID",
            result.run_id[:8] + "...",
            help=f"Full run ID: {result.run_id}",
        )

    with col2:
        stage = result.stage.value.replace("_", " ").title()
        st.metric("Stage", stage)

    with col3:
        st.metric("Candidates", len(result.candidates))

    with col4:
        st.metric("Warnings", len(result.warnings))

    # Timing information
    if "total_duration_seconds" in metadata:
        duration = metadata["total_duration_seconds"]
        st.metric(
            "Total Duration",
            f"{duration:.2f}s",
            help="Total pipeline execution time",
        )

    # Stage timing breakdown
    if "stage_timing" in metadata:
        with st.expander("⏱️ Stage Timing Details"):
            timing_data = []
            for stage_name, timing in metadata["stage_timing"].items():
                timing_data.append({
                    "Stage": stage_name.replace("_", " ").title(),
                    "Duration (s)": f"{timing['duration_seconds']:.3f}",
                })

            if timing_data:
                df = pd.DataFrame(timing_data)
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_molecular_properties(result: DrugDiscoveryResult) -> None:
    """Render molecular properties section."""
    st.markdown('<div class="section-header">🧪 Molecular Properties</div>', unsafe_allow_html=True)

    candidates_with_props = [
        c for c in result.candidates
        if c.molecular_properties is not None
    ]

    if not candidates_with_props:
        st.warning("No molecular properties available. Check if the pipeline completed all stages.")
        return

    data = []
    for candidate in candidates_with_props:
        props = candidate.molecular_properties
        data.append({
            "Compound": candidate.compound.name,
            "MW (g/mol)": format_molecular_weight_display(props.molecular_weight),
            "LogP": f"{props.logp:.2f}",
            "TPSA": f"{props.tpsa:.1f}",
            "H-Donors": props.h_bond_donors,
            "H-Acceptors": props.h_bond_acceptors,
            "Rotatable Bonds": props.rotatable_bonds,
            "Aromatic Rings": props.aromatic_rings,
            "Lipinski Violations": props.lipinski_violations,
            "QED": f"{props.qed_score:.3f}" if props.qed_score is not None else "N/A",
            "Lipinski Compliant": "✅" if props.is_lipinski_compliant() else "❌",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_similarity_results(result: DrugDiscoveryResult) -> None:
    """Render compound similarity results."""
    st.markdown('<div class="section-header">🔍 Similarity Search Results</div>', unsafe_allow_html=True)

    similar_data = []
    
    # Check if we have similarity data in metadata
    if "similarity_scores" in result.metadata:
        for compound_id, score in result.metadata["similarity_scores"].items():
            candidate = next(
                (c for c in result.candidates if c.compound.compound_id == compound_id),
                None
            )
            name = candidate.compound.name if candidate else compound_id
            similar_data.append({
                "Compound": name,
                "Similarity Score": format_similarity_display(score),
                "Rank": len(similar_data) + 1,
            })
    else:
        # Try to get from context - check if any candidate has similarity info
        for candidate in result.candidates:
            # Check if candidate has similarity in its metadata
            if hasattr(candidate, 'similarity_score') and candidate.similarity_score is not None:
                similar_data.append({
                    "Compound": candidate.compound.name,
                    "Similarity Score": format_similarity_display(candidate.similarity_score),
                    "Rank": len(similar_data) + 1,
                })

    if not similar_data:
        st.info("No similarity search results available.")
        return

    df = pd.DataFrame(similar_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_target_predictions(result: DrugDiscoveryResult) -> None:
    """Render target prediction results."""
    st.markdown('<div class="section-header">🎯 Target Predictions</div>', unsafe_allow_html=True)

    candidates_with_targets = [
        c for c in result.candidates
        if c.target_predictions
    ]

    if not candidates_with_targets:
        st.info("No target predictions available.")
        return

    for candidate in candidates_with_targets[:5]:
        with st.expander(f"🎯 {candidate.compound.name} ({len(candidate.target_predictions)} targets)"):
            data = []
            for pred in sorted(candidate.target_predictions, key=lambda x: x.confidence_score, reverse=True):
                data.append({
                    "Target": pred.target.name,
                    "Type": pred.target.target_type.value,
                    "Confidence": format_confidence(pred.confidence_score),
                    "Level": pred.confidence_level.value.upper(),
                    "Method": pred.prediction_method or "N/A",
                })

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_toxicity_predictions(result: DrugDiscoveryResult) -> None:
    """Render toxicity prediction results."""
    st.markdown('<div class="section-header">☣️ Toxicity Predictions</div>', unsafe_allow_html=True)

    candidates_with_toxicity = [
        c for c in result.candidates
        if c.toxicity_prediction is not None
    ]

    if not candidates_with_toxicity:
        st.info("No toxicity predictions available.")
        return

    data = []
    for candidate in candidates_with_toxicity:
        pred = candidate.toxicity_prediction
        toxicity_class = get_toxicity_class(pred.toxicity_level)

        data.append({
            "Compound": candidate.compound.name,
            "Toxicity Level": f'<span class="{toxicity_class}">{pred.toxicity_level.value.upper()}</span>',
            "Toxicity Score": f"{pred.toxicity_score:.3f}",
            "Confidence": pred.confidence.value.upper(),
            "Endpoints": ", ".join(pred.endpoint_scores.keys()) if pred.endpoint_scores else "None",
        })

    df = pd.DataFrame(data)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    for candidate in candidates_with_toxicity:
        pred = candidate.toxicity_prediction
        with st.expander(f"📋 Detailed Toxicity: {candidate.compound.name}"):
            if pred.endpoint_scores:
                endpoint_data = []
                for endpoint, score in pred.endpoint_scores.items():
                    endpoint_data.append({
                        "Endpoint": endpoint.replace("_", " ").title(),
                        "Score": f"{score:.3f}",
                        "Risk Level": "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low",
                    })
                df_endpoints = pd.DataFrame(endpoint_data)
                st.dataframe(df_endpoints, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{pred.toxicity_score:.3f}")
            with col2:
                st.metric("Level", pred.toxicity_level.value.upper())
            with col3:
                st.metric("Confidence", pred.confidence.value.upper())


def render_repurposing_results(result: DrugDiscoveryResult) -> None:
    """Render drug repurposing results."""
    st.markdown('<div class="section-header">💊 Drug Repurposing</div>', unsafe_allow_html=True)

    if "repurposing_candidates" not in result.metadata:
        st.info("No repurposing results available.")
        return

    repurposing_data = result.metadata.get("repurposing_candidates", [])

    if not repurposing_data:
        st.info("No repurposing candidates found.")
        return

    data = []
    for item in repurposing_data:
        data.append({
            "Compound": item.get("name", "Unknown"),
            "Score": f"{item.get('score', 0):.3f}",
            "Confidence": item.get("confidence", "N/A"),
            "Shared Targets": item.get("shared_targets", 0),
            "KG Relationships": item.get("kg_relationships", 0),
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_candidate_ranking(result: DrugDiscoveryResult) -> None:
    """Render candidate ranking results."""
    st.markdown('<div class="section-header">🏆 Candidate Ranking</div>', unsafe_allow_html=True)

    if not result.rankings:
        st.info("No ranking results available.")
        return

    data = []
    for ranking in result.rankings[:10]:
        candidate = next(
            (c for c in result.candidates if c.candidate_id == ranking.candidate_id),
            None
        )
        if candidate:
            data.append({
                "Rank": ranking.rank,
                "Compound": candidate.compound.name,
                "Score": f"{ranking.composite_score:.3f}",
                "Confidence": ranking.notes.split("Confidence:")[1].split(".")[0] if ranking.notes and "Confidence:" in ranking.notes else "N/A",
                "Status": candidate.status.value.replace("_", " ").title() if candidate.status else "Pending",
            })

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    for ranking in result.rankings[:5]:
        candidate = next(
            (c for c in result.candidates if c.candidate_id == ranking.candidate_id),
            None
        )
        if candidate:
            with st.expander(f"📊 Rank #{ranking.rank}: {candidate.compound.name} (Score: {ranking.composite_score:.3f})"):
                if ranking.criteria_scores:
                    criteria_data = []
                    for criterion, score in ranking.criteria_scores.items():
                        criteria_data.append({
                            "Criterion": criterion.replace("_", " ").title(),
                            "Score": f"{score:.3f}",
                        })
                    df_criteria = pd.DataFrame(criteria_data)
                    st.dataframe(df_criteria, use_container_width=True, hide_index=True)

                if ranking.notes:
                    st.caption(f"**Notes:** {ranking.notes}")


def render_full_report(result: DrugDiscoveryResult) -> None:
    """Render a full text report."""
    with st.expander("📄 Full Report", expanded=False):
        lines = [
            "# Drug Discovery Report",
            "",
            f"**Run ID:** {result.run_id}",
            f"**Stage:** {result.stage.value.replace('_', ' ').title()}",
            f"**Created:** {format_iso_timestamp(result.created_at)}",
            f"**Completed:** {format_iso_timestamp(result.completed_at) if result.completed_at else 'N/A'}",
            f"**Candidates:** {len(result.candidates)}",
            "",
            "## Candidates",
        ]

        for idx, candidate in enumerate(result.candidates, 1):
            lines.append(f"### {idx}. {candidate.compound.name}")
            lines.append(f"- **ID:** {candidate.compound.compound_id}")
            lines.append(f"- **SMILES:** {candidate.compound.smiles}")
            if candidate.molecular_properties:
                props = candidate.molecular_properties
                lines.append(f"- **MW:** {props.molecular_weight:.2f} g/mol")
                lines.append(f"- **LogP:** {props.logp:.2f}")
                lines.append(f"- **Lipinski Violations:** {props.lipinski_violations}")
            if candidate.overall_score is not None:
                lines.append(f"- **Score:** {candidate.overall_score:.3f}")
            if candidate.notes:
                lines.append(f"- **Notes:** {candidate.notes}")
            lines.append("")

        if result.warnings:
            lines.extend([
                "",
                "## Warnings",
                *[f"- {w}" for w in result.warnings],
            ])

        report_text = "\n".join(lines)
        st.text_area("Report", report_text, height=400)


# ----------------------------------------------------------------------
# Export functions
# ----------------------------------------------------------------------

def export_to_json(result: DrugDiscoveryResult) -> str:
    """Export result as JSON."""
    data = {
        "run_id": result.run_id,
        "stage": result.stage.value,
        "created_at": format_iso_timestamp(result.created_at),
        "completed_at": format_iso_timestamp(result.completed_at) if result.completed_at else None,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "compound": {
                    "compound_id": c.compound.compound_id,
                    "name": c.compound.name,
                    "smiles": c.compound.smiles,
                },
                "overall_score": c.overall_score,
                "status": c.status.value if c.status else None,
                "notes": c.notes,
            }
            for c in result.candidates
        ],
        "rankings": [
            {
                "rank": r.rank,
                "candidate_id": r.candidate_id,
                "composite_score": r.composite_score,
                "criteria_scores": r.criteria_scores,
                "notes": r.notes,
            }
            for r in result.rankings
        ],
        "warnings": result.warnings,
        "metadata": result.metadata,
    }
    return json.dumps(data, indent=2, default=str)


def export_to_csv(result: DrugDiscoveryResult) -> str:
    """Export result as CSV."""
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Rank", "Candidate ID", "Compound Name", "SMILES",
        "Overall Score", "Status", "MW (g/mol)", "LogP",
        "Lipinski Violations", "Toxicity Level", "Toxicity Score"
    ])

    for ranking in sorted(result.rankings, key=lambda r: r.rank):
        candidate = next(
            (c for c in result.candidates if c.candidate_id == ranking.candidate_id),
            None
        )
        if not candidate:
            continue

        props = candidate.molecular_properties
        tox = candidate.toxicity_prediction

        writer.writerow([
            ranking.rank,
            candidate.candidate_id,
            candidate.compound.name,
            candidate.compound.smiles,
            f"{ranking.composite_score:.3f}",
            candidate.status.value if candidate.status else "pending",
            f"{props.molecular_weight:.2f}" if props else "",
            f"{props.logp:.2f}" if props else "",
            props.lipinski_violations if props else "",
            tox.toxicity_level.value if tox else "",
            f"{tox.toxicity_score:.3f}" if tox else "",
        ])

    return output.getvalue()


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------

def main() -> None:
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = ""
    if "disease_name_input" not in st.session_state:
        st.session_state.disease_name_input = ""
    if "drug_discovery_results" not in st.session_state:
        st.session_state.drug_discovery_results = None
    if "drug_discovery_loading" not in st.session_state:
        st.session_state.drug_discovery_loading = False
    """Main application entry point."""
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🧬 BioNexus Drug Discovery</h1>
        <p>AI-powered drug discovery and repurposing platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    compounds, query_smiles, disease_name = render_input_section()

    # Show placeholder when no input and no results
    if compounds is None and not st.session_state.result:
        st.info("👆 Enter a SMILES string and click 'Run Drug Discovery Analysis' to begin.")
        return

    # Run pipeline
    if compounds:
        if st.session_state.is_running:
            st.warning("Analysis is already running...")
            return

        # Execute pipeline
        try:
            st.session_state.is_running = True

            # Show progress
            progress_bar = st.progress(0, text="Initializing...")

            # Build configuration
            config = PipelineConfig(
                top_k=st.session_state.top_k,
                enable_cache=st.session_state.enable_cache,
                enable_similarity=st.session_state.enable_similarity,
                enable_target_prediction=st.session_state.enable_target_prediction,
                enable_toxicity_prediction=st.session_state.enable_toxicity_prediction,
                enable_repurposing=st.session_state.enable_repurposing,
            )
            config.discovery_config.similarity_threshold = st.session_state.similarity_threshold

            # Update progress
            progress_bar.progress(20, text="Running drug discovery pipeline...")

            # Execute pipeline
            result = run_drug_discovery(
                compounds=compounds,
                query_smiles=query_smiles,
                disease_name=disease_name,
                config=config,
            )

            # Store result
            st.session_state.result = result

            # Update progress
            progress_bar.progress(80, text="Processing results...")

            # Clear progress
            progress_bar.progress(100, text="Complete!")

            # Success message
            st.success(f"✅ Analysis complete! Found {len(result.candidates)} candidates.")

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.exception("Pipeline execution failed")

        finally:
            st.session_state.is_running = False

    # Display results if available
    if st.session_state.result:
        result = st.session_state.result

        # Show summary info
        st.markdown("---")
        st.markdown("### 📋 Analysis Summary")
        st.markdown(f"**Run ID:** `{result.run_id}`")
        st.markdown(f"**Stage:** `{result.stage.value}`")
        st.markdown(f"**Candidates found:** `{len(result.candidates)}`")

        # Show candidate names
        if result.candidates:
            st.markdown("#### Candidate Compounds")
            for i, candidate in enumerate(result.candidates[:5]):
                score = candidate.overall_score if candidate.overall_score is not None else "N/A"
                st.write(f"**{i+1}.** {candidate.compound.name} - Score: {score}")

            if len(result.candidates) > 5:
                st.write(f"... and {len(result.candidates) - 5} more")

        # Show stage results for debugging
        if "stage_results" in result.metadata:
            with st.expander("🔍 Pipeline Stage Execution Details"):
                for stage_name, stage_data in result.metadata["stage_results"].items():
                    status = "✅" if stage_data.get("success") else "❌" if not stage_data.get("skipped") else "⏭️"
                    error = f" - Error: {stage_data.get('error')}" if stage_data.get('error') else ""
                    st.write(f"{status} **{stage_name}**: {stage_data.get('data', {})}{error}")

        # Execution stats
        render_execution_stats(result)

        # Results sections
        tabs = st.tabs([
            "🧪 Properties",
            "🔍 Similarity",
            "🎯 Targets",
            "☣️ Toxicity",
            "💊 Repurposing",
            "🏆 Ranking",
            "📄 Full Report",
        ])

        with tabs[0]:
            render_molecular_properties(result)

        with tabs[1]:
            render_similarity_results(result)

        with tabs[2]:
            render_target_predictions(result)

        with tabs[3]:
            render_toxicity_predictions(result)

        with tabs[4]:
            render_repurposing_results(result)

        with tabs[5]:
            render_candidate_ranking(result)

        with tabs[6]:
            render_full_report(result)

        # Export section
        st.markdown("---")
        st.markdown("### 📥 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            json_data = export_to_json(result)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"drug_discovery_{result.run_id}.json",
                mime="application/json",
            )

        with col2:
            csv_data = export_to_csv(result)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"drug_discovery_{result.run_id}.csv",
                mime="text/csv",
            )

        with col3:
            report_text = f"# Drug Discovery Report\n\n**Run ID:** {result.run_id}\n\n"
            report_text += f"**Candidates:** {len(result.candidates)}\n\n"
            for ranking in result.rankings[:10]:
                candidate = next(
                    (c for c in result.candidates if c.candidate_id == ranking.candidate_id),
                    None
                )
                if candidate:
                    report_text += f"**{ranking.rank}. {candidate.compound.name}** - Score: {ranking.composite_score:.3f}\n\n"

            st.download_button(
                label="📝 Download Report",
                data=report_text,
                file_name=f"drug_discovery_{result.run_id}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()