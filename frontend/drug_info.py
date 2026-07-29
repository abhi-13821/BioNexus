"""
frontend/drug_info.py

BioNexus — Drug Information Module
------------------------------------
An AI-powered biomedical drug discovery assistant module that retrieves,
analyzes, and visually presents comprehensive chemical/pharmacological
data for a searched compound using the public PubChem REST API and
PubChemPy.

This module exposes a single public entry point:

    def show() -> None

which is intended to be imported and called from the main Streamlit
application (app.py). No other symbols should be relied upon externally.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

try:
    import pubchempy as pcp
except ImportError:  # pragma: no cover - defensive guard
    pcp = None


# ==========================================================================
# EMBEDDINGS INTEGRATION (BioNexus semantic search)
# ==========================================================================
# Reuses the existing, already-tested `embeddings` package as-is: no changes
# to embedding_generator.py, embedding_store.py, embedding_search.py,
# embedding_manager.py, cache.py, models.py, or utils.py are made or needed.
# Imports are deferred into `_get_embedding_manager()` (lazy loading) so that
# `import frontend.drug_info` never pays the cost of loading the embedding
# model or its dependencies unless a drug lookup actually happens.

# Where drug embeddings are persisted — kept separate from
# frontend/literature_search.py's own embeddings store, since drugs and
# papers are different kinds of records.
_DRUG_EMBEDDINGS_STORAGE_PATH = "data/embeddings/drug_info_embeddings.json"

# source_type values distinguish the three embedded "facets" of a compound
# within the shared embeddings store, so semantic search can be scoped to
# just one facet (e.g. only mechanism-of-action text) instead of mixing them.
_DRUG_SOURCE_TYPE_GENERAL = "drug_general"
_DRUG_SOURCE_TYPE_MOA = "drug_moa"
_DRUG_SOURCE_TYPE_INDICATION = "drug_indication"

_DEFAULT_SIMILAR_DRUGS_TOP_K = 5

# Lightweight keyword heuristics used to mine mechanism-of-action and
# indication-specific sentences out of PubChem's free-text description,
# mirroring the dependency-free heuristic approach already used in
# frontend/literature_search.py (extract_entities), rather than introducing
# a new NLP dependency for this module.
_MOA_KEYWORDS = (
    "inhibit", "inhibitor", "inhibits", "inhibiting", "agonist", "antagonist",
    "receptor", "enzyme", "block", "blocks", "blocking", "bind", "binds",
    "binding", "activates", "activation", "mechanism", "pathway", "kinase",
    "channel", "transporter", "substrate", "catalyzes", "modulates",
)
_INDICATION_KEYWORDS = (
    "used to treat", "indicated for", "treatment of", "used for",
    "prevention of", "management of", "therapy for", "relief of",
    "used in the treatment", "approved for", "used against", "treats",
)


@st.cache_resource(show_spinner=False)
def _get_embedding_manager():
    """
    Build (once per Streamlit server process) the EmbeddingManager used to
    turn drug descriptions into vector embeddings for semantic similarity
    search and drug recommendations.

    Wrapped in ``st.cache_resource`` so the underlying sentence-transformers
    model, the on-disk embedding store, and the in-memory embedding cache
    are all constructed exactly once per server process rather than on
    every Streamlit rerun. Imports are performed inside this function
    (lazy loading) so modules that never call it never pay the import cost.

    Returns
    -------
    EmbeddingManager
        A manager wired to the default
        ``sentence-transformers/all-MiniLM-L6-v2`` generator and a
        JSON-backed store dedicated to drug embeddings.
    """
    from embeddings.embedding_generator import SentenceTransformerEmbeddingGenerator
    from embeddings.embedding_manager import EmbeddingManager
    from embeddings.embedding_store import JSONEmbeddingStore

    generator = SentenceTransformerEmbeddingGenerator()
    store = JSONEmbeddingStore(storage_path=_DRUG_EMBEDDINGS_STORAGE_PATH)
    return EmbeddingManager(generator=generator, store=store)


# ==========================================================================
# CONSTANTS
# ==========================================================================

PUBCHEM_PUG_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_COMPOUND_PAGE = "https://pubchem.ncbi.nlm.nih.gov/compound"
PUBCHEM_BIOASSAY_PAGE = "https://www.ncbi.nlm.nih.gov/pcassay?term="
PUBCHEM_PATENT_PAGE = "https://pubchem.ncbi.nlm.nih.gov/compound"

REQUEST_TIMEOUT_SECONDS = 15
MAX_SYNONYMS_DISPLAYED = 20
EXAMPLE_DRUGS = ["Aspirin", "Ibuprofen", "Paracetamol", "Metformin", "Caffeine"]

PROPERTY_FIELDS = (
    "IUPACName,CanonicalSMILES,IsomericSMILES,MolecularFormula,"
    "MolecularWeight,ExactMass,MonoisotopicMass,XLogP,TPSA,"
    "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,"
    "HeavyAtomCount,Charge,Complexity"
)

LIPINSKI_MW_LIMIT = 500
LIPINSKI_LOGP_LIMIT = 5
LIPINSKI_HBD_LIMIT = 5
LIPINSKI_HBA_LIMIT = 10

# Colors (dark research theme)
COLOR_BG = "#0b0f19"
COLOR_CARD_BG = "#131826"
COLOR_CARD_BORDER = "#1f2937"
COLOR_ACCENT = "#3b82f6"
COLOR_ACCENT_LIGHT = "#60a5fa"
COLOR_TEXT = "#e5e7eb"
COLOR_TEXT_MUTED = "#9ca3af"
COLOR_PASS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_FAIL = "#ef4444"


# ==========================================================================
# DATA MODELS
# ==========================================================================

@dataclass
class CompoundProperties:
    """Structured container for PubChem compound properties."""

    cid: Optional[int] = None
    iupac_name: str = "N/A"
    canonical_smiles: str = "N/A"
    isomeric_smiles: str = "N/A"
    molecular_formula: str = "N/A"
    molecular_weight: Optional[float] = None
    exact_mass: Optional[float] = None
    monoisotopic_mass: Optional[float] = None
    xlogp: Optional[float] = None
    tpsa: Optional[float] = None
    hbond_donor_count: Optional[int] = None
    hbond_acceptor_count: Optional[int] = None
    rotatable_bond_count: Optional[int] = None
    heavy_atom_count: Optional[int] = None
    formal_charge: Optional[int] = None
    complexity: Optional[float] = None
    synonyms: List[str] = field(default_factory=list)
    description: str = "No description available for this compound."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# STYLING
# ==========================================================================

def _inject_custom_css() -> None:
    """Inject custom dark-themed CSS for a professional research dashboard look."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
        }}

        .bx-header {{
            padding: 1.5rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, {COLOR_CARD_BG} 0%, #0f1729 100%);
            border: 1px solid {COLOR_CARD_BORDER};
            margin-bottom: 1.5rem;
        }}
        .bx-header h1 {{
            color: {COLOR_TEXT};
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }}
        .bx-header p {{
            color: {COLOR_TEXT_MUTED};
            margin-top: 0.35rem;
            font-size: 0.95rem;
        }}

        .bx-card {{
            background-color: {COLOR_CARD_BG};
            border: 1px solid {COLOR_CARD_BORDER};
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }}
        .bx-card h3 {{
            color: {COLOR_ACCENT_LIGHT};
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid {COLOR_CARD_BORDER};
            padding-bottom: 0.5rem;
        }}
        .bx-card p, .bx-card li {{
            color: {COLOR_TEXT};
            font-size: 0.92rem;
            line-height: 1.55;
        }}

        .bx-badge {{
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-right: 0.4rem;
        }}
        .bx-badge-pass {{
            background-color: rgba(16, 185, 129, 0.15);
            color: {COLOR_PASS};
            border: 1px solid {COLOR_PASS};
        }}
        .bx-badge-warning {{
            background-color: rgba(245, 158, 11, 0.15);
            color: {COLOR_WARNING};
            border: 1px solid {COLOR_WARNING};
        }}
        .bx-badge-fail {{
            background-color: rgba(239, 68, 68, 0.15);
            color: {COLOR_FAIL};
            border: 1px solid {COLOR_FAIL};
        }}

        .bx-insight {{
            background-color: rgba(59, 130, 246, 0.08);
            border-left: 3px solid {COLOR_ACCENT};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
            color: {COLOR_TEXT};
        }}
        .bx-insight b {{
            color: {COLOR_ACCENT_LIGHT};
        }}

        .bx-link-btn {{
            display: inline-block;
            padding: 0.5rem 1.1rem;
            margin-right: 0.6rem;
            margin-bottom: 0.5rem;
            border-radius: 10px;
            background-color: rgba(59, 130, 246, 0.12);
            border: 1px solid {COLOR_ACCENT};
            color: {COLOR_ACCENT_LIGHT} !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 0.85rem;
            transition: all 0.15s ease-in-out;
        }}
        .bx-link-btn:hover {{
            background-color: {COLOR_ACCENT};
            color: #fff !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: {COLOR_CARD_BG};
            border: 1px solid {COLOR_CARD_BORDER};
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }}

        .stTextInput input {{
            background-color: {COLOR_CARD_BG} !important;
            color: {COLOR_TEXT} !important;
            border: 1px solid {COLOR_CARD_BORDER} !important;
            border-radius: 10px !important;
        }}

        .stButton button {{
            background-color: {COLOR_ACCENT};
            color: #fff;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }}
        .stButton button:hover {{
            background-color: {COLOR_ACCENT_LIGHT};
            color: #0b0f19;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# DATA RETRIEVAL (CACHED)
# ==========================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_cid(query: str) -> Optional[int]:
    """Resolve a drug/compound name to a PubChem CID."""
    if pcp is None:
        return None
    try:
        compounds = pcp.get_compounds(query, "name")
        if compounds and compounds[0].cid:
            return int(compounds[0].cid)
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_properties_json(cid: int) -> Optional[Dict[str, Any]]:
    """Fetch numeric/structural properties for a CID via the PUG REST API."""
    url = (
        f"{PUBCHEM_PUG_REST_BASE}/compound/cid/{cid}/property/"
        f"{PROPERTY_FIELDS}/JSON"
    )
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        return props[0] if props else None
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_synonyms(cid: int) -> List[str]:
    """Fetch synonyms for a given CID."""
    url = f"{PUBCHEM_PUG_REST_BASE}/compound/cid/{cid}/synonyms/JSON"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        info = data.get("InformationList", {}).get("Information", [])
        if info:
            return info[0].get("Synonym", [])[:MAX_SYNONYMS_DISPLAYED]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return []


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_description(cid: int) -> str:
    """Fetch a textual description/summary for a given CID."""
    url = f"{PUBCHEM_PUG_REST_BASE}/compound/cid/{cid}/description/JSON"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        entries = data.get("InformationList", {}).get("Information", [])
        for entry in entries:
            if "Description" in entry:
                return entry["Description"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return "No description available for this compound."


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_structure_image_bytes(cid: int) -> Optional[bytes]:
    """Fetch the official 2D PubChem structure image (PNG) for a CID."""
    url = f"{PUBCHEM_PUG_REST_BASE}/compound/cid/{cid}/PNG?record_type=2d&image_size=large"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def _as_float(value: Any) -> Optional[float]:
    """Safely coerce a PubChem property value (often returned as str) to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """Safely coerce a PubChem property value (often returned as str) to int."""
    parsed = _as_float(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (TypeError, ValueError):
        return None


def fetch_compound_properties(query: str) -> Optional[CompoundProperties]:
    """
    Orchestrate a full lookup: resolve CID, then aggregate properties,
    synonyms, and description into a single CompoundProperties object.
    """
    cid = _fetch_cid(query)
    if cid is None:
        return None

    raw_props = _fetch_properties_json(cid) or {}
    synonyms = _fetch_synonyms(cid)
    description = _fetch_description(cid)

    return CompoundProperties(
        cid=cid,
        iupac_name=raw_props.get("IUPACName") or "N/A",
        canonical_smiles=raw_props.get("CanonicalSMILES") or "N/A",
        isomeric_smiles=raw_props.get("IsomericSMILES") or "N/A",
        molecular_formula=raw_props.get("MolecularFormula") or "N/A",
        molecular_weight=_as_float(raw_props.get("MolecularWeight")),
        exact_mass=_as_float(raw_props.get("ExactMass")),
        monoisotopic_mass=_as_float(raw_props.get("MonoisotopicMass")),
        xlogp=_as_float(raw_props.get("XLogP")),
        tpsa=_as_float(raw_props.get("TPSA")),
        hbond_donor_count=_as_int(raw_props.get("HBondDonorCount")),
        hbond_acceptor_count=_as_int(raw_props.get("HBondAcceptorCount")),
        rotatable_bond_count=_as_int(raw_props.get("RotatableBondCount")),
        heavy_atom_count=_as_int(raw_props.get("HeavyAtomCount")),
        formal_charge=_as_int(raw_props.get("Charge")),
        complexity=_as_float(raw_props.get("Complexity")),
        synonyms=synonyms,
        description=description if description else "No description available for this compound.",
    )


# ==========================================================================
# EMBEDDINGS INTEGRATION — metadata conversion, extraction, ingestion
# ==========================================================================

def _extract_sentences_by_keywords(text: str, keywords: tuple) -> str:
    """
    Extract sentences from `text` that contain at least one of `keywords`
    (case-insensitive substring match), joined back into a single string.

    A lightweight, dependency-free heuristic — no NLP library required —
    used to mine mechanism-of-action- and indication-specific text out of
    PubChem's free-text compound description.

    Parameters
    ----------
    text:
        The source text to mine (typically `CompoundProperties.description`).
    keywords:
        Keyword/phrase substrings to match against, case-insensitively.

    Returns
    -------
    str
        The matched sentences joined with a space, or an empty string if
        no sentence matched or `text` is empty.
    """
    if not text:
        return ""

    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    lowered_keywords = [k.lower() for k in keywords]
    matched = [
        sentence for sentence in sentences
        if any(keyword in sentence.lower() for keyword in lowered_keywords)
    ]
    return " ".join(matched).strip()


def _is_usable_description(text: str) -> bool:
    """Return whether `text` is non-empty and not a known placeholder string."""
    if not text:
        return False
    normalized = text.strip().lower()
    return normalized not in ("", "no description available for this compound.")


def _build_drug_embedding_metadata_list(props: CompoundProperties, drug_name: str) -> list:
    """
    Build the list of `EmbeddingMetadata` records representing a compound,
    one per embedded "facet": general description, mechanism-of-action
    text, and indication text.

    Facets with no usable text are omitted (e.g. if PubChem has no
    description at all, or no MOA/indication-flavored sentences were
    found within it), so we never store a meaningless placeholder
    embedding.

    Parameters
    ----------
    props:
        The compound's fetched properties.
    drug_name:
        The human-entered search term (e.g. "Aspirin"), used as the
        display title for the embedding records — friendlier than the
        often highly technical IUPAC name.

    Returns
    -------
    list[EmbeddingMetadata]
        Zero to three metadata records ready for embedding and storage.
    """
    from embeddings.models import EmbeddingMetadata

    if props.cid is None or not _is_usable_description(props.description):
        return []

    source_id = str(props.cid)
    # Preserve the complete compound record on every facet so any one of
    # them is sufficient to reconstruct a displayable result later.
    extra = props.to_dict()
    extra["drug_name"] = drug_name

    moa_text = _extract_sentences_by_keywords(props.description, _MOA_KEYWORDS)
    indication_text = _extract_sentences_by_keywords(props.description, _INDICATION_KEYWORDS)

    facets = [
        (_DRUG_SOURCE_TYPE_GENERAL, props.description, 0),
    ]
    if moa_text:
        facets.append((_DRUG_SOURCE_TYPE_MOA, moa_text, 1))
    if indication_text:
        facets.append((_DRUG_SOURCE_TYPE_INDICATION, indication_text, 2))

    return [
        EmbeddingMetadata(
            source_id=source_id,
            source_type=source_type,
            title=drug_name,
            text=facet_text,
            chunk_index=chunk_index,
            extra=extra,
        )
        for source_type, facet_text, chunk_index in facets
    ]


def _reconstruct_compound_from_metadata(metadata) -> tuple:
    """
    Rebuild a displayable `(CompoundProperties, drug_name)` pair from a
    stored `EmbeddingMetadata` record.

    Parameters
    ----------
    metadata:
        A metadata record previously produced by
        `_build_drug_embedding_metadata_list`.

    Returns
    -------
    tuple[CompoundProperties, str]
        The reconstructed compound properties and its display name.
    """
    extra = dict(metadata.extra or {})
    drug_name = extra.pop("drug_name", metadata.title)
    known_fields = {f for f in CompoundProperties.__dataclass_fields__}
    filtered = {k: v for k, v in extra.items() if k in known_fields}
    return CompoundProperties(**filtered), drug_name


def ingest_drug_embeddings(props: CompoundProperties, drug_name: str) -> Dict[str, int]:
    """
    Generate and store embeddings for a compound, reusing existing
    embeddings instead of regenerating them whenever this compound (by
    PubChem CID) has already been indexed.

    Never raises: failures are logged via Streamlit's session state is not
    touched here, and all errors are caught so a single embedding failure
    can never interrupt the rest of the drug info page from rendering.

    Parameters
    ----------
    props:
        The compound's fetched properties.
    drug_name:
        The human-entered search term, used as the display title.

    Returns
    -------
    dict[str, int]
        Summary counters: ``embedded`` (facets newly embedded),
        ``skipped_duplicate`` (1 if this compound was already indexed, 0
        otherwise), ``skipped_no_text``, and ``failed``.
    """
    stats = {"embedded": 0, "skipped_duplicate": 0, "skipped_no_text": 0, "failed": 0}

    if props.cid is None:
        stats["skipped_no_text"] = 1
        return stats

    try:
        manager = _get_embedding_manager()
    except Exception:
        stats["failed"] = 1
        return stats

    try:
        already_indexed = manager.get_by_source(str(props.cid))
    except Exception:
        already_indexed = []

    if already_indexed:
        stats["skipped_duplicate"] = 1
        return stats

    metadata_list = _build_drug_embedding_metadata_list(props, drug_name)
    if not metadata_list:
        stats["skipped_no_text"] = 1
        return stats

    for metadata in metadata_list:
        try:
            manager.add_document(metadata)
            stats["embedded"] += 1
        except Exception:
            stats["failed"] += 1

    return stats


def _search_similar_by_facet(
    props: CompoundProperties,
    source_type: str,
    facet_text: str,
    top_k: int,
) -> list:
    """
    Shared implementation backing `find_similar_drugs`,
    `find_similar_by_mechanism`, and `find_similar_by_indication`: run a
    semantic search scoped to a single facet's `source_type`, excluding
    the queried compound's own records from the results.

    Reuses the manager's embedding cache: `facet_text` is typically
    identical to text already embedded and stored for this compound, so
    this call is served from cache rather than re-invoking the model.

    Parameters
    ----------
    props:
        The compound being queried against (excluded from its own results).
    source_type:
        Which embedded facet to search within (general / MOA / indication).
    facet_text:
        The text to use as the semantic query (e.g. the compound's own
        description, MOA text, or indication text).
    top_k:
        Maximum number of other compounds to return.

    Returns
    -------
    list[SimilaritySearchResult]
        Ranked results, most similar first, excluding the queried compound.
    """
    if not facet_text or props.cid is None:
        return []

    try:
        manager = _get_embedding_manager()
        # Request one extra result so we still have `top_k` after
        # excluding the queried compound itself (which will normally be
        # its own closest match).
        raw_results = manager.search(facet_text, top_k=top_k + 1, source_type=source_type)
    except Exception:
        return []

    own_source_id = str(props.cid)
    filtered = [r for r in raw_results if r.record.metadata.source_id != own_source_id]
    return filtered[:top_k]


def find_similar_drugs(
    props: CompoundProperties, top_k: int = _DEFAULT_SIMILAR_DRUGS_TOP_K
) -> list:
    """
    Recommend drugs overall semantically similar to `props`, based on
    full compound description similarity.

    Parameters
    ----------
    props:
        The compound to find similar drugs for.
    top_k:
        Maximum number of recommendations to return.

    Returns
    -------
    list[SimilaritySearchResult]
        Ranked results, most similar first.
    """
    return _search_similar_by_facet(props, _DRUG_SOURCE_TYPE_GENERAL, props.description, top_k)


def find_similar_by_mechanism(
    props: CompoundProperties, top_k: int = _DEFAULT_SIMILAR_DRUGS_TOP_K
) -> list:
    """
    Recommend drugs with a similar mechanism of action to `props`, based
    on mechanism-flavored sentences mined from the compound description.

    Returns an empty list if no mechanism-of-action text could be
    identified for this compound (e.g. its description doesn't mention
    receptors, inhibition, binding, etc.).

    Parameters
    ----------
    props:
        The compound to find mechanism-of-action matches for.
    top_k:
        Maximum number of recommendations to return.

    Returns
    -------
    list[SimilaritySearchResult]
        Ranked results, most similar first.
    """
    moa_text = _extract_sentences_by_keywords(props.description, _MOA_KEYWORDS)
    return _search_similar_by_facet(props, _DRUG_SOURCE_TYPE_MOA, moa_text, top_k)


def find_similar_by_indication(
    props: CompoundProperties, top_k: int = _DEFAULT_SIMILAR_DRUGS_TOP_K
) -> list:
    """
    Recommend drugs with a similar indication (therapeutic use) to
    `props`, based on indication-flavored sentences mined from the
    compound description.

    Returns an empty list if no indication text could be identified for
    this compound.

    Parameters
    ----------
    props:
        The compound to find indication matches for.
    top_k:
        Maximum number of recommendations to return.

    Returns
    -------
    list[SimilaritySearchResult]
        Ranked results, most similar first.
    """
    indication_text = _extract_sentences_by_keywords(props.description, _INDICATION_KEYWORDS)
    return _search_similar_by_facet(props, _DRUG_SOURCE_TYPE_INDICATION, indication_text, top_k)


def semantic_drug_search(query_text: str, top_k: int = _DEFAULT_SIMILAR_DRUGS_TOP_K) -> list:
    """
    Search previously-indexed drugs by natural-language meaning rather
    than exact keyword/name matching.

    Parameters
    ----------
    query_text:
        A free-text description of what the user is looking for, e.g.
        "a pain reliever that also reduces fever".
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[SimilaritySearchResult]
        Ranked results, most similar first. Empty if `query_text` is
        blank or the search fails for any reason.
    """
    if not query_text or not query_text.strip():
        return []
    try:
        manager = _get_embedding_manager()
        return manager.search(query_text, top_k=top_k, source_type=_DRUG_SOURCE_TYPE_GENERAL)
    except Exception:
        return []


# ==========================================================================
# DRUG-LIKENESS (LIPINSKI RULE OF FIVE)
# ==========================================================================

def evaluate_lipinski(props: CompoundProperties) -> Dict[str, Any]:
    """
    Evaluate Lipinski's Rule of Five against compound properties.

    Returns a dict containing individual rule checks, number of violations,
    and an overall verdict of PASS / WARNING / FAIL.
    """
    checks = {
        "Molecular Weight <= 500": (
            props.molecular_weight is not None
            and props.molecular_weight <= LIPINSKI_MW_LIMIT
        ),
        "LogP <= 5": (
            props.xlogp is not None and props.xlogp <= LIPINSKI_LOGP_LIMIT
        ),
        "H-Bond Donors <= 5": (
            props.hbond_donor_count is not None
            and props.hbond_donor_count <= LIPINSKI_HBD_LIMIT
        ),
        "H-Bond Acceptors <= 10": (
            props.hbond_acceptor_count is not None
            and props.hbond_acceptor_count <= LIPINSKI_HBA_LIMIT
        ),
    }

    violations = sum(1 for passed in checks.values() if not passed)

    if violations == 0:
        verdict = "PASS"
    elif violations == 1:
        verdict = "WARNING"
    else:
        verdict = "FAIL"

    return {"checks": checks, "violations": violations, "verdict": verdict}


def _render_badge(verdict: str) -> str:
    """Return HTML markup for a colored PASS/WARNING/FAIL badge."""
    css_class = {
        "PASS": "bx-badge-pass",
        "WARNING": "bx-badge-warning",
        "FAIL": "bx-badge-fail",
    }.get(verdict, "bx-badge-warning")
    return f'<span class="bx-badge {css_class}">{verdict}</span>'


# ==========================================================================
# MOLECULAR INSIGHTS (EXPLANATIONS)
# ==========================================================================

def _generate_molecular_insights(props: CompoundProperties) -> List[str]:
    """Generate simple, human-readable biomedical explanations of key properties."""
    insights = []

    if props.molecular_weight is not None:
        insights.append(
            f"<b>Molecular Weight ({props.molecular_weight:.2f} g/mol):</b> "
            "Reflects the size of the molecule. Smaller molecules "
            "(under 500 g/mol) generally absorb and distribute in the body "
            "more easily."
        )
    if props.xlogp is not None:
        insights.append(
            f"<b>LogP ({props.xlogp:.2f}):</b> Measures how well the compound "
            "dissolves in fats versus water. Higher values mean the molecule "
            "is more fat-soluble, which affects how easily it crosses cell "
            "membranes, including the blood-brain barrier."
        )
    if props.tpsa is not None:
        insights.append(
            f"<b>TPSA ({props.tpsa:.2f} \u00c5\u00b2):</b> Topological Polar "
            "Surface Area estimates the molecule's ability to permeate "
            "cell membranes. Lower TPSA values are generally associated "
            "with better absorption and membrane permeability."
        )
    if props.hbond_donor_count is not None:
        insights.append(
            f"<b>Hydrogen Bond Donors ({props.hbond_donor_count}):</b> "
            "Atoms that can donate hydrogen bonds, influencing how the "
            "molecule interacts with biological targets and water."
        )
    if props.hbond_acceptor_count is not None:
        insights.append(
            f"<b>Hydrogen Bond Acceptors ({props.hbond_acceptor_count}):</b> "
            "Atoms that can accept hydrogen bonds, affecting solubility and "
            "the strength of binding to biological receptors."
        )

    return insights


# ==========================================================================
# EXPORT HELPERS
# ==========================================================================

def _build_json_export(props: CompoundProperties) -> bytes:
    return json.dumps(props.to_dict(), indent=2).encode("utf-8")


def _build_txt_report(props: CompoundProperties, lipinski: Dict[str, Any]) -> bytes:
    lines = [
        "BioNexus — Drug Information Report",
        "=" * 40,
        f"PubChem CID: {props.cid}",
        f"IUPAC Name: {props.iupac_name}",
        f"Molecular Formula: {props.molecular_formula}",
        f"Molecular Weight: {props.molecular_weight}",
        f"Exact Mass: {props.exact_mass}",
        f"Monoisotopic Mass: {props.monoisotopic_mass}",
        f"Canonical SMILES: {props.canonical_smiles}",
        f"Isomeric SMILES: {props.isomeric_smiles}",
        f"XLogP: {props.xlogp}",
        f"TPSA: {props.tpsa}",
        f"H-Bond Donors: {props.hbond_donor_count}",
        f"H-Bond Acceptors: {props.hbond_acceptor_count}",
        f"Rotatable Bonds: {props.rotatable_bond_count}",
        f"Heavy Atom Count: {props.heavy_atom_count}",
        f"Formal Charge: {props.formal_charge}",
        f"Complexity: {props.complexity}",
        "",
        f"Lipinski Verdict: {lipinski['verdict']} "
        f"({lipinski['violations']} violation(s))",
        "",
        "Description:",
        props.description,
        "",
        "Synonyms:",
        ", ".join(props.synonyms) if props.synonyms else "N/A",
    ]
    return "\n".join(lines).encode("utf-8")


def _build_csv_export(props: CompoundProperties) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Property", "Value"])
    for key, value in props.to_dict().items():
        if key == "synonyms":
            value = "; ".join(value) if value else ""
        writer.writerow([key, value])
    return buffer.getvalue().encode("utf-8")


# ==========================================================================
# UI RENDERING HELPERS
# ==========================================================================

def _render_header() -> None:
    st.markdown(
        """
        <div class="bx-header">
            <h1>🧬 Drug Information</h1>
            <p>Search any drug, compound, or chemical name to retrieve
            structured biomedical and chemical data from PubChem.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_search_bar() -> str:
    col_input, col_examples = st.columns([3, 2])

    # Render example buttons FIRST and, on click, stash the choice in a
    # separate session_state key before the text_input widget (which owns
    # "drug_info_query_input") is instantiated. Streamlit forbids writing to
    # a widget's own key after that widget has been created in the same run,
    # so we use a rerun to apply the pending value cleanly on the next run.
    with col_examples:
        st.caption("Quick examples")
        example_cols = st.columns(len(EXAMPLE_DRUGS))
        for col, name in zip(example_cols, EXAMPLE_DRUGS):
            if col.button(name, key=f"example_{name}"):
                st.session_state["drug_info_query_input"] = name
                st.rerun()

    with col_input:
        query = st.text_input(
            "Search a drug, compound, or chemical name",
            placeholder="e.g. Aspirin, Ibuprofen, Metformin...",
            key="drug_info_query_input",
        )

    return query.strip() if query else ""


def _render_structure_card(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>🧪 2D Structure</h3>', unsafe_allow_html=True)
    image_bytes = _fetch_structure_image_bytes(props.cid) if props.cid else None
    if image_bytes:
        st.image(image_bytes, caption=f"CID {props.cid}", use_container_width=False)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        st.markdown(
            f'<a class="bx-link-btn" href="data:image/png;base64,{b64_image}" '
            f'download="{props.iupac_name or "structure"}.png">⬇ Download Structure Image</a>',
            unsafe_allow_html=True,
        )
        st.caption("Tip: click the image to expand/zoom via the built-in viewer.")
    else:
        st.warning("Structure image unavailable for this compound.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_property_dashboard(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>📊 Property Dashboard</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Molecular Weight", f"{props.molecular_weight:.2f}" if props.molecular_weight else "N/A")
    col2.metric("XLogP", f"{props.xlogp:.2f}" if props.xlogp is not None else "N/A")
    col3.metric("TPSA", f"{props.tpsa:.2f}" if props.tpsa is not None else "N/A")
    col4.metric("Complexity", f"{props.complexity:.1f}" if props.complexity is not None else "N/A")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("H-Bond Donors", props.hbond_donor_count if props.hbond_donor_count is not None else "N/A")
    col6.metric("H-Bond Acceptors", props.hbond_acceptor_count if props.hbond_acceptor_count is not None else "N/A")
    col7.metric("Rotatable Bonds", props.rotatable_bond_count if props.rotatable_bond_count is not None else "N/A")
    col8.metric("Heavy Atoms", props.heavy_atom_count if props.heavy_atom_count is not None else "N/A")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_identity_card(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>🔬 Compound Identity</h3>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p><b>PubChem CID:</b> {props.cid}</p>
        <p><b>IUPAC Name:</b> {props.iupac_name}</p>
        <p><b>Molecular Formula:</b> {props.molecular_formula}</p>
        <p><b>Canonical SMILES:</b> <code>{props.canonical_smiles}</code></p>
        <p><b>Isomeric SMILES:</b> <code>{props.isomeric_smiles}</code></p>
        <p><b>Exact Mass:</b> {props.exact_mass}</p>
        <p><b>Monoisotopic Mass:</b> {props.monoisotopic_mass}</p>
        <p><b>Formal Charge:</b> {props.formal_charge}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_description_card(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>📝 Description &amp; Known Uses</h3>', unsafe_allow_html=True)
    st.markdown(f"<p>{props.description}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_synonyms_card(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>🏷️ Synonyms</h3>', unsafe_allow_html=True)
    if props.synonyms:
        with st.expander(f"Show first {len(props.synonyms)} synonyms"):
            for synonym in props.synonyms:
                st.markdown(f"- {synonym}")
    else:
        st.caption("No synonyms found.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_lipinski_card(props: CompoundProperties) -> Dict[str, Any]:
    lipinski = evaluate_lipinski(props)
    st.markdown('<div class="bx-card"><h3>💊 Drug-Likeness — Lipinski Rule of Five</h3>', unsafe_allow_html=True)
    st.markdown(
        f"Overall Verdict: {_render_badge(lipinski['verdict'])} "
        f"&nbsp;({lipinski['violations']} violation(s))",
        unsafe_allow_html=True,
    )
    for rule, passed in lipinski["checks"].items():
        badge = _render_badge("PASS") if passed else _render_badge("FAIL")
        st.markdown(f"{badge} {rule}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return lipinski


def _render_insights_card(props: CompoundProperties) -> None:
    st.markdown('<div class="bx-card"><h3>🧠 Molecular Insights</h3>', unsafe_allow_html=True)
    insights = _generate_molecular_insights(props)
    if insights:
        for insight in insights:
            st.markdown(f'<div class="bx-insight">{insight}</div>', unsafe_allow_html=True)
    else:
        st.caption("Insufficient data to generate insights.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_downloads_card(props: CompoundProperties, lipinski: Dict[str, Any]) -> None:
    st.markdown('<div class="bx-card"><h3>⬇️ Export Data</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download JSON",
        data=_build_json_export(props),
        file_name=f"{props.iupac_name or 'compound'}_bionexus.json",
        mime="application/json",
    )
    col2.download_button(
        "Download TXT Report",
        data=_build_txt_report(props, lipinski),
        file_name=f"{props.iupac_name or 'compound'}_bionexus_report.txt",
        mime="text/plain",
    )
    col3.download_button(
        "Download CSV",
        data=_build_csv_export(props),
        file_name=f"{props.iupac_name or 'compound'}_bionexus.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_similarity_score_badge(score: float) -> str:
    """Return HTML markup for a similarity-score badge, colored by strength."""
    if score >= 0.75:
        css_class = "bx-badge-pass"
    elif score >= 0.5:
        css_class = "bx-badge-warning"
    else:
        css_class = "bx-badge-fail"
    return f'<span class="bx-badge {css_class}">{score:.3f}</span>'


def _render_recommendation_card(title: str, icon: str, results: list, empty_message: str) -> None:
    """
    Render a single recommendation card (Similar Drugs / Similar Mechanism
    of Action / Similar Indications) from a list of SimilaritySearchResult.

    Parameters
    ----------
    title:
        Card heading text (without icon).
    icon:
        Emoji icon prefixed to the heading.
    results:
        The ranked `SimilaritySearchResult` list to display.
    empty_message:
        Message shown when `results` is empty.
    """
    st.markdown(f'<div class="bx-card"><h3>{icon} {title}</h3>', unsafe_allow_html=True)
    if not results:
        st.caption(empty_message)
    else:
        for result in results:
            try:
                other_props, other_name = _reconstruct_compound_from_metadata(result.record.metadata)
            except Exception:
                continue
            score_badge = _render_similarity_score_badge(result.score)
            st.markdown(
                f"<p><b>{other_name}</b> &nbsp;{score_badge} "
                f"&nbsp;<span style='color:{COLOR_TEXT_MUTED};'>"
                f"(CID {other_props.cid}, {other_props.molecular_formula})</span></p>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_similar_drugs_section(props: CompoundProperties, top_k: int = _DEFAULT_SIMILAR_DRUGS_TOP_K) -> None:
    """
    Render the "Similar Drugs" recommendation cards: overall semantic
    similarity, similar mechanism of action, and similar indications.

    All lookups are read-only against the embeddings store and degrade
    gracefully to an informative empty state if the store has too few
    other compounds indexed yet, or if this compound has no usable
    description text for a given facet.
    """
    similar_drugs = find_similar_drugs(props, top_k=top_k)
    _render_recommendation_card(
        "Similar Drugs (Semantic Similarity)", "🔗", similar_drugs,
        "No semantically similar drugs indexed yet. Search a few more "
        "compounds to build up the semantic index.",
    )

    similar_moa = find_similar_by_mechanism(props, top_k=top_k)
    _render_recommendation_card(
        "Similar Mechanism of Action", "⚙️", similar_moa,
        "No mechanism-of-action text could be identified for this compound, "
        "or no other indexed compounds share one yet.",
    )

    similar_indications = find_similar_by_indication(props, top_k=top_k)
    _render_recommendation_card(
        "Similar Indications", "🎯", similar_indications,
        "No indication (therapeutic use) text could be identified for this "
        "compound, or no other indexed compounds share one yet.",
    )


def _render_semantic_search_section() -> None:
    """
    Render a collapsible natural-language semantic search section, letting
    users search previously-indexed drugs by meaning rather than exact
    name/keyword — e.g. "a pain reliever that also reduces fever".

    Collapsed by default so the existing exact-name search flow and
    layout are unaffected for users who don't open it.
    """
    with st.expander("🧠 Semantic Drug Search (search by meaning, not exact name)"):
        st.caption(
            "Search previously indexed drugs using a natural-language "
            "description instead of an exact drug name."
        )
        with st.form("drug_info_semantic_search_form"):
            semantic_query = st.text_input(
                "Describe what you're looking for",
                placeholder="e.g. a drug that reduces inflammation and pain",
                key="drug_info_semantic_query_input",
            )
            top_k = st.slider("Number of results (Top-K)", min_value=1, max_value=20, value=5,
                               key="drug_info_semantic_top_k")
            submitted = st.form_submit_button("Search by meaning")

        if not submitted:
            return

        if not semantic_query or not semantic_query.strip():
            st.error("Please enter a description to search for.")
            return

        try:
            manager = _get_embedding_manager()
            indexed_count = manager.count()
        except Exception as exc:
            st.error(f"Semantic search is currently unavailable: {exc}")
            return

        if indexed_count == 0:
            st.warning(
                "No drugs are indexed yet. Search for a few drugs by name "
                "first — results are automatically embedded and added to "
                "the semantic index."
            )
            return

        results = semantic_drug_search(semantic_query, top_k=top_k)
        if not results:
            st.info("No semantically similar drugs found for that description.")
            return

        st.success(f"Found {len(results)} semantically matching drug(s).")
        for result in results:
            try:
                other_props, other_name = _reconstruct_compound_from_metadata(result.record.metadata)
            except Exception:
                continue
            score_badge = _render_similarity_score_badge(result.score)
            st.markdown(
                f"**{other_name}** &nbsp;{score_badge} "
                f"(CID {other_props.cid}, {other_props.molecular_formula})",
                unsafe_allow_html=True,
            )
            st.caption(other_props.description)


# ==========================================================================
# ERROR HANDLING
# ==========================================================================

def _check_connectivity() -> bool:
    """Perform a lightweight connectivity check against PubChem."""
    try:
        requests.get(
            f"{PUBCHEM_PUG_REST_BASE}/compound/cid/1/property/MolecularFormula/JSON",
            timeout=5,
        )
        return True
    except requests.RequestException:
        return False


def _render_error(message: str) -> None:
    st.markdown(
        f"""
        <div class="bx-card" style="border-color: {COLOR_FAIL};">
            <h3 style="color:{COLOR_FAIL};">⚠️ Error</h3>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# PUBLIC ENTRY POINT
# ==========================================================================

def show() -> None:
    """
    Render the Drug Information module.

    This is the single public entry point expected by the main BioNexus
    Streamlit application.
    """
    _inject_custom_css()
    _render_header()
    _render_semantic_search_section()

    if pcp is None:
        _render_error(
            "The PubChemPy library is not installed. Please install it "
            "via <code>pip install pubchempy</code> to use this module."
        )
        return

    query = _render_search_bar()

    if not query:
        st.info("Enter a drug, compound, or chemical name above to begin, "
                 "or select one of the quick examples.")
        return

    if len(query) < 2:
        _render_error("Please enter a valid, non-trivial search query.")
        return

    try:
        with st.spinner(f"Searching PubChem for '{query}'..."):
            props = fetch_compound_properties(query)
    except requests.exceptions.Timeout:
        _render_error("The request to PubChem timed out. Please try again shortly.")
        return
    except requests.exceptions.ConnectionError:
        _render_error(
            "Unable to reach PubChem. Please check your internet connection "
            "and try again."
        )
        return
    except Exception as exc:  # pragma: no cover - defensive catch-all
        _render_error(f"An unexpected error occurred: {exc}")
        return

    if props is None:
        if not _check_connectivity():
            _render_error(
                "PubChem appears to be unreachable right now. This could be "
                "due to no internet connection or PubChem service downtime. "
                "Please try again later."
            )
        else:
            _render_error(
                f"No results found for <b>'{query}'</b>. Please check the "
                "spelling or try a different drug/compound/chemical name."
            )
        return

    st.success(f"Found compound: **{props.iupac_name}** (CID: {props.cid})")

    col_left, col_right = st.columns([1, 1.4])
    with col_left:
        _render_structure_card(props)
    with col_right:
        _render_identity_card(props)
        _render_property_dashboard(props)

    _render_description_card(props)
    _render_synonyms_card(props)

    lipinski = _render_lipinski_card(props)
    _render_insights_card(props)
    _render_downloads_card(props, lipinski)

    try:
        with st.spinner("Updating semantic drug index…"):
            ingest_drug_embeddings(props, query)
        _render_similar_drugs_section(props)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        st.caption(f"Similar-drug recommendations are temporarily unavailable: {exc}")