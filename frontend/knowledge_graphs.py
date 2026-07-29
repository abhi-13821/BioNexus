"""
frontend/knowledge_graphs.py
==============================================================================
BioNexus — Biomedical Knowledge Graph Explorer
==============================================================================

A self-contained Streamlit module that turns the papers found by the
Literature Search module (frontend/literature_search.py) into a small,
human-readable biomedical knowledge graph.

Pipeline
--------
    st.session_state.bx_last_results (list[Paper], set by literature_search.py)
            |
            v
    knowledge_graph.kg_pipeline.KnowledgeGraphPipeline   (entity + relation
            |                                              extraction, not
            v                                              reimplemented here)
    raw NetworkX graph (thousands of nodes)
            |
            v
    intelligent filtering + centrality/frequency ranking   (this module)
            |
            v
    Top-N nodes / Top-M edges  ->  PyVis interactive visualization + table
            |
            v
    embeddings.EmbeddingManager  ->  semantic search / clustering / similarity
                                      / layout / expansion / Q&A   (this module)

This module never modifies the raw graph produced by the backend pipeline
and never re-implements entity/relation extraction — it only filters, ranks
and renders what the pipeline already produced. Likewise, it never
re-implements embedding generation, storage, caching, or similarity search —
it only calls into the existing, already-tested ``embeddings`` package via
``EmbeddingManager``.

Usage
-----
    # app.py
    from frontend import knowledge_graphs
    knowledge_graphs.show()
"""

from __future__ import annotations

import os
import re
import logging
import builtins
import tempfile
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger("bionexus.knowledge_graphs")
logging.basicConfig(level=logging.INFO)

# --- Optional / backend dependencies, imported defensively -------------------
try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    _HAS_NETWORKX = False

try:
    from knowledge_graph.kg_pipeline import KnowledgeGraphPipeline
    _HAS_PIPELINE = True
except ImportError:  # pragma: no cover
    _HAS_PIPELINE = False

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except ImportError:  # pragma: no cover
    _HAS_PYVIS = False

# --- EMBEDDINGS INTEGRATION ---------------------------------------------------
# Reuse the existing, already-implemented and already-tested `embeddings`
# package (EmbeddingManager + supporting types) rather than duplicating any
# embedding generation, storage, caching, or similarity-search logic here.
# This module only calls into that public API.
try:
    from embeddings.embedding_manager import EmbeddingManager, EmbeddingManagerError
    from embeddings.embedding_generator import SentenceTransformerEmbeddingGenerator
    from embeddings.embedding_store import JSONEmbeddingStore
    from embeddings.models import EmbeddingMetadata
    from embeddings.utils import cosine_similarity, compute_text_hash
    _HAS_EMBEDDINGS = True
except ImportError:  # pragma: no cover
    _HAS_EMBEDDINGS = False
# --- END EMBEDDINGS INTEGRATION (imports) -------------------------------------


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class KGConfig:
    """Central configuration for the knowledge graph module."""

    ENTITY_MODEL = "en_core_sci_sm"

    # Node text filtering
    MAX_NODE_LABEL_LEN = 50
    MAX_NODE_WORDS = 6            # more words than this looks like a sentence fragment

    # Hard caps (requirement: never show more than this, regardless of sliders)
    TOP_NODES_CAP = 50
    TOP_EDGES_CAP = 100

    # UI slider defaults
    DEFAULT_MAX_NODES = 30
    DEFAULT_MAX_EDGES = 50

    # Ranking weights
    DEGREE_WEIGHT = 0.6
    FREQUENCY_WEIGHT = 0.4

    # Visual style per entity type (label colors)
    TYPE_COLORS = {
        "disease": "#e74c3c",     # red
        "gene": "#3498db",        # blue
        "drug": "#2ecc71",        # green
        "chemical": "#2ecc71",    # treated like drug
        "protein": "#9b59b6",     # purple
        "default": "#95a5a6",     # grey fallback
    }

    TYPE_LEGEND = [
        ("Disease", "#e74c3c"),
        ("Gene", "#3498db"),
        ("Drug / Chemical", "#2ecc71"),
        ("Protein", "#9b59b6"),
        ("Other / Unknown", "#95a5a6"),
    ]

    # --- EMBEDDINGS INTEGRATION: configuration for semantic features --------
    # Storage location for knowledge-graph entity/relation embeddings. Kept
    # separate from other modules' embedding stores (e.g. literature_search's)
    # so this module never collides with or overwrites their data.
    EMBEDDINGS_STORAGE_PATH = os.path.join("data", "embeddings", "kg_embeddings.json")

    # source_type tags used when storing entity vs. relation embeddings, so
    # semantic search can be scoped to just one or the other.
    ENTITY_SOURCE_TYPE = "kg_entity"
    RELATION_SOURCE_TYPE = "kg_relation"

    DEFAULT_SIMILAR_NODES_TOP_K = 8
    DEFAULT_CLUSTER_THRESHOLD = 0.75
    DEFAULT_EXPANSION_TOP_K = 8
    # --- END EMBEDDINGS INTEGRATION (configuration) -------------------------


# A compact, dependency-free stopword list — enough to catch the kind of
# noise (pronouns, conjunctions, generic verbs) that NER pipelines commonly
# leak into node lists.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "we", "you",
    "they", "he", "she", "it", "i", "our", "their", "its", "his", "her",
    "not", "no", "yes", "do", "does", "did", "have", "has", "had", "can",
    "could", "will", "would", "should", "may", "might", "must", "than",
    "also", "into", "onto", "than", "such", "via", "per", "vs", "about",
    "between", "among", "within", "study", "studies", "result", "results",
    "method", "methods", "conclusion", "conclusions", "background",
    "objective", "objectives", "purpose", "aim", "aims",
}

_HTML_PATTERN = re.compile(r"<[^>]+>")
_STAT_PATTERN = re.compile(
    r"(p\s*[<=>]\s*0?\.\d+)"                 # p=0.001, p<0.05
    r"|(\d+(\.\d+)?\s*%\s*(ci|confidence))"  # 95% confidence interval
    r"|(\bci\b\s*[:=]?\s*\d)"                # CI: 1.2
    r"|(\bn\s*=\s*\d+)"                      # n=120
    r"|(\bhr\s*[:=]\s*\d)"                   # HR=1.5
    r"|(\bor\s*[:=]\s*\d)",                  # OR=2.1
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"^[\d\.\-\+%,/]+$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


# ==============================================================================
# 2. NODE / TEXT FILTERING
# ==============================================================================

def _clean_text(text: Any) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(text or "")).strip()


def is_meaningful_node(label: Any) -> bool:
    """Decide whether a raw extracted node label is a genuine biomedical
    concept worth displaying, as opposed to noise (stop words, statistics,
    HTML fragments, numbers, or full sentences)."""
    text = _clean_text(label)

    if not text:
        return False
    if len(text) > KGConfig.MAX_NODE_LABEL_LEN:
        return False
    if _HTML_PATTERN.search(text):
        return False
    if _STAT_PATTERN.search(text.lower()):
        return False
    if _NUMBER_PATTERN.match(text):
        return False
    if not re.search(r"[A-Za-z]", text):
        return False

    words = text.split()

    # Single common word / stopword / single letter -> noise
    if len(words) == 1:
        lowered = text.lower().strip(".,;:()[]")
        if lowered in _STOPWORDS:
            return False
        if len(lowered) <= 2 and not lowered.isupper():
            # allow short uppercase gene/protein symbols like "P53" but not "we"
            return False

    # Long word sequences are usually sentence fragments, not concepts
    if len(words) > KGConfig.MAX_NODE_WORDS:
        return False

    # Reject strings that are mostly punctuation/symbols
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < max(2, len(text) * 0.4):
        return False

    return True


# ==============================================================================
# 3. GRAPH ATTRIBUTE HELPERS
# ==============================================================================
# The exact schema produced by knowledge_graph.graph_builder isn't assumed to
# be fixed, so these helpers look for several plausible attribute names and
# fall back gracefully.

def _node_type(data: dict) -> str:
    for key in ("type", "entity_type", "label_type", "category", "ner_type"):
        val = data.get(key)
        if val:
            return str(val).strip().lower()
    return "default"


def _node_color(node_type: str) -> str:
    return KGConfig.TYPE_COLORS.get(node_type, KGConfig.TYPE_COLORS["default"])


def _node_weight(data: dict) -> float:
    for key in ("count", "frequency", "freq", "weight", "occurrences"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1.0


def _edge_label(data: dict) -> str:
    for key in ("relation", "relation_type", "label", "predicate", "type"):
        val = data.get(key)
        if val:
            return str(val)
    return "related_to"


def _edge_weight(data: dict) -> float:
    for key in ("weight", "count", "frequency", "score"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1.0


# ==============================================================================
# 4. FILTERING + RANKING PIPELINE
# ==============================================================================

def _build_filtered_graph(
    raw_graph: "nx.Graph",
    max_nodes: int,
    max_edges: int,
    min_importance: float,
) -> tuple["nx.DiGraph", dict[str, float]]:
    """Reduce a large raw extraction graph down to a small, ranked subgraph
    that a human can actually read.

    Steps: drop noisy node labels -> score remaining nodes by a blend of
    degree centrality and occurrence frequency -> keep the top-N nodes
    (never exceeding the module's hard cap) -> keep the top-M edges between
    those nodes (again capped).
    """
    empty = nx.DiGraph()

    if raw_graph is None or raw_graph.number_of_nodes() == 0:
        return empty, {}

    working = raw_graph if raw_graph.is_directed() else raw_graph.to_directed()

    # Step 1 — drop noisy nodes
    valid_nodes = [n for n in working.nodes() if is_meaningful_node(n)]
    if not valid_nodes:
        return empty, {}

    sub = working.subgraph(valid_nodes).copy()
    if sub.number_of_nodes() == 0:
        return empty, {}

    # Step 2 — importance = blend of degree centrality and frequency
    try:
        degree_centrality = nx.degree_centrality(sub)
    except Exception:
        degree_centrality = {n: 0.0 for n in sub.nodes()}

    freq = {n: _node_weight(data) for n, data in sub.nodes(data=True)}
    max_freq = max(freq.values()) if freq else 1.0
    max_freq = max_freq or 1.0

    importance: dict[str, float] = {}
    for n in sub.nodes():
        norm_freq = freq.get(n, 1.0) / max_freq
        importance[n] = (
            KGConfig.DEGREE_WEIGHT * degree_centrality.get(n, 0.0)
            + KGConfig.FREQUENCY_WEIGHT * norm_freq
        )

    # Step 3 — apply minimum importance filter, then keep top-N nodes
    effective_max_nodes = min(max_nodes, KGConfig.TOP_NODES_CAP)
    candidates = [n for n, score in importance.items() if score >= min_importance]
    if not candidates:
        candidates = list(importance.keys())

    ranked_nodes = sorted(candidates, key=lambda n: importance[n], reverse=True)
    top_nodes = ranked_nodes[:effective_max_nodes]

    node_subgraph = sub.subgraph(top_nodes).copy()

    # Step 4 — rank and keep top-M edges among the surviving nodes
    effective_max_edges = min(max_edges, KGConfig.TOP_EDGES_CAP)
    scored_edges = []
    for u, v, data in node_subgraph.edges(data=True):
        score = importance.get(u, 0.0) + importance.get(v, 0.0) + _edge_weight(data)
        scored_edges.append((u, v, data, score))
    scored_edges.sort(key=lambda item: item[3], reverse=True)
    top_edges = scored_edges[:effective_max_edges]

    result = nx.DiGraph()
    for n in top_nodes:
        attrs = dict(node_subgraph.nodes[n])
        attrs["importance"] = importance.get(n, 0.0)
        result.add_node(n, **attrs)
    for u, v, data, _score in top_edges:
        result.add_edge(u, v, **data)

    # Drop nodes that ended up with no edges at all AND low importance, so
    # isolated noise doesn't clutter the view — but keep at least a handful
    # of high-importance isolated concepts if that's all there is.
    if result.number_of_edges() > 0:
        connected = {n for edge in result.edges() for n in edge}
        isolated_high_value = sorted(
            (n for n in result.nodes() if n not in connected),
            key=lambda n: importance.get(n, 0.0),
            reverse=True,
        )[:5]
        keep = connected | set(isolated_high_value)
        result = result.subgraph(keep).copy()

    return result, importance


# ==============================================================================
# 4b. EMBEDDINGS INTEGRATION — manager lifecycle, sync, and semantic features
# ==============================================================================
# Everything in this section only *calls* the existing `embeddings` package
# (EmbeddingManager / EmbeddingMetadata / cosine_similarity / etc.). No
# embedding generation, storage, or search logic is reimplemented here.

@st.cache_resource(show_spinner=False)
def _get_embedding_manager() -> Optional["EmbeddingManager"]:
    """Lazily construct a singleton EmbeddingManager for this module.

    Cached via ``st.cache_resource`` so the underlying model, LRU cache, and
    JSON-backed store are built once per app process (lazy loading) and
    reused across reruns/interactions, rather than being reconstructed (and
    losing the in-memory cache) on every Streamlit rerun.
    """
    if not _HAS_EMBEDDINGS:
        return None
    try:
        generator = SentenceTransformerEmbeddingGenerator()
        store = JSONEmbeddingStore(storage_path=KGConfig.EMBEDDINGS_STORAGE_PATH)
        return EmbeddingManager(generator=generator, store=store)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize EmbeddingManager: %s", exc)
        return None


def _entity_source_id(label: str) -> str:
    """Deterministic, content-addressed source_id for an entity embedding.

    Using a hash of the label (rather than a random ID) means the same
    entity encountered across separate graph builds maps to the same
    ``source_id``, which is what lets us look up and reuse an existing
    embedding instead of regenerating it.
    """
    return f"kg_entity::{compute_text_hash(label)}"


def _relation_source_id(subject: str, relation: str, obj: str) -> str:
    """Deterministic source_id for a (subject, relation, object) triple."""
    return f"kg_relation::{compute_text_hash(f'{subject}|{relation}|{obj}')}"


def _sync_entity_embeddings(
    graph: "nx.DiGraph", manager: "EmbeddingManager"
) -> dict[str, str]:
    """Ensure every node in `graph` has a corresponding stored embedding.

    For each node, first checks whether an embedding already exists (via its
    deterministic source_id) and reuses it if so; only genuinely new entities
    are embedded, and those are embedded in a single batch call (rather than
    one-by-one) for efficiency.

    Returns
    -------
    dict[str, str]
        Mapping of node label -> embedding_id, for every node that has (or
        now has) a stored embedding.
    """
    node_to_embedding_id: dict[str, str] = {}
    pending_metadata: list["EmbeddingMetadata"] = []
    pending_labels: list[str] = []

    for node, data in graph.nodes(data=True):
        label = str(node)
        source_id = _entity_source_id(label)

        # Reuse: skip regeneration if this entity was already embedded
        # (e.g. in a previous graph build, or by another module).
        existing = manager.get_by_source(source_id)
        if existing:
            node_to_embedding_id[label] = existing[0].embedding_id
            continue

        entity_type = _node_type(data)
        metadata = EmbeddingMetadata(
            source_id=source_id,
            source_type=KGConfig.ENTITY_SOURCE_TYPE,
            title=label,
            text=f"{label} ({entity_type})" if entity_type != "default" else label,
            extra={"entity_type": entity_type},
        )
        pending_metadata.append(metadata)
        pending_labels.append(label)

    if pending_metadata:
        try:
            records = manager.add_documents(pending_metadata)
            for label, record in zip(pending_labels, records):
                node_to_embedding_id[label] = record.embedding_id
            logger.info(
                "Generated %d new entity embedding(s); reused %d existing.",
                len(records),
                len(node_to_embedding_id) - len(records),
            )
        except (EmbeddingManagerError, ValueError) as exc:
            logger.warning("Batch entity embedding generation failed: %s", exc)

    return node_to_embedding_id


def _sync_relation_embeddings(
    graph: "nx.DiGraph", manager: "EmbeddingManager"
) -> dict[tuple[str, str, str], str]:
    """Ensure every important relation (edge) in `graph` has a stored
    embedding, reusing existing ones and batching new ones, exactly as
    :func:`_sync_entity_embeddings` does for nodes.

    Returns
    -------
    dict[(subject, relation, object), str]
        Mapping of the relation triple -> embedding_id.
    """
    edge_to_embedding_id: dict[tuple[str, str, str], str] = {}
    pending_metadata: list["EmbeddingMetadata"] = []
    pending_keys: list[tuple[str, str, str]] = []

    for u, v, data in graph.edges(data=True):
        subject, obj = str(u), str(v)
        relation = _edge_label(data)
        key = (subject, relation, obj)
        source_id = _relation_source_id(subject, relation, obj)

        existing = manager.get_by_source(source_id)
        if existing:
            edge_to_embedding_id[key] = existing[0].embedding_id
            continue

        metadata = EmbeddingMetadata(
            source_id=source_id,
            source_type=KGConfig.RELATION_SOURCE_TYPE,
            title=relation,
            text=f"{subject} {relation} {obj}",
            extra={"subject": subject, "relation": relation, "object": obj},
        )
        pending_metadata.append(metadata)
        pending_keys.append(key)

    if pending_metadata:
        try:
            records = manager.add_documents(pending_metadata)
            for key, record in zip(pending_keys, records):
                edge_to_embedding_id[key] = record.embedding_id
            logger.info(
                "Generated %d new relation embedding(s); reused %d existing.",
                len(records),
                len(edge_to_embedding_id) - len(records),
            )
        except (EmbeddingManagerError, ValueError) as exc:
            logger.warning("Batch relation embedding generation failed: %s", exc)

    return edge_to_embedding_id


def _compute_embedding_layout(
    manager: "EmbeddingManager", node_to_embedding_id: dict[str, str]
) -> dict[str, tuple[float, float]]:
    """Project node embeddings into 2D so semantically similar nodes can be
    placed closer together in the visualization.

    Uses a lightweight, dependency-free PCA (via ``numpy.linalg.svd``) over
    the vectors already produced by the embeddings module — no new ML
    dependency is introduced, and no existing layout code is removed: the
    result is only used as an optional *initial* position hint for PyVis,
    which still runs its existing physics simulation on top of it.

    Returns an empty dict (meaning: fall back to the existing physics-only
    layout) if there are too few embedded nodes to project meaningfully, or
    if anything goes wrong.
    """
    labels: list[str] = []
    vectors: list[list[float]] = []
    for label, embedding_id in node_to_embedding_id.items():
        try:
            record = manager.get(embedding_id)
        except EmbeddingManagerError:
            continue
        labels.append(label)
        vectors.append(record.vector)

    if len(vectors) < 2:
        return {}

    try:
        matrix = np.asarray(vectors, dtype=float)
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        components = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt[0], vt[0]])
        coords_2d = matrix @ components.T

        max_abs = float(np.abs(coords_2d).max()) or 1.0
        scale = 400.0
        coords_2d = coords_2d * (scale / max_abs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding-based layout projection failed: %s", exc)
        return {}

    return {label: (float(x), float(y)) for label, (x, y) in zip(labels, coords_2d)}


def _cluster_entities_by_embedding(
    manager: "EmbeddingManager",
    node_to_embedding_id: dict[str, str],
    threshold: float,
) -> list[list[str]]:
    """Group semantically similar entities using cosine similarity over
    their existing embeddings (a simple greedy single-linkage clustering —
    no new ML dependency is introduced; this reuses
    ``embeddings.utils.cosine_similarity`` directly).

    Parameters
    ----------
    threshold:
        Minimum cosine similarity for two entities to be placed in the same
        cluster.

    Returns
    -------
    list[list[str]]
        Each inner list is one cluster of entity labels.
    """
    vectors: dict[str, list[float]] = {}
    for label, embedding_id in node_to_embedding_id.items():
        try:
            vectors[label] = manager.get(embedding_id).vector
        except EmbeddingManagerError:
            continue

    labels = list(vectors.keys())
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for label in labels:
        if label in assigned:
            continue
        cluster = [label]
        assigned.add(label)
        for other in labels:
            if other in assigned:
                continue
            try:
                score = cosine_similarity(vectors[label], vectors[other])
            except ValueError:
                continue
            if score >= threshold:
                cluster.append(other)
                assigned.add(other)
        clusters.append(cluster)

    return clusters


# --- Semantic UI sections ------------------------------------------------

def _render_semantic_node_search(
    manager: "EmbeddingManager", node_to_embedding_id: dict[str, str]
) -> None:
    """Free-text semantic search over graph entities (e.g. 'lung cancer protein')."""
    st.caption(
        "Search graph entities by meaning instead of exact name — e.g. "
        "*\"lung cancer protein\"*."
    )
    query = st.text_input("Search nodes", key="kg_node_search_query")
    if not query:
        return
    if not node_to_embedding_id:
        st.info("No embedded entities available yet.")
        return
    try:
        results = manager.search(query, top_k=10, source_type=KGConfig.ENTITY_SOURCE_TYPE)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Semantic node search failed: {exc}")
        return
    if not results:
        st.info("No matching entities found.")
        return
    df = pd.DataFrame(
        [
            {
                "Entity": r.record.metadata.title,
                "Type": r.record.metadata.extra.get("entity_type", "unknown"),
                "Relevance": round(r.score, 3),
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_semantic_relation_search(manager: "EmbeddingManager") -> None:
    """Free-text semantic search over graph relations (e.g. 'drugs treating diabetes')."""
    st.caption(
        "Search relationships by meaning — e.g. *\"proteins causing disease\"* "
        "or *\"drugs treating diabetes\"*."
    )
    query = st.text_input("Search relations", key="kg_relation_search_query")
    if not query:
        return
    try:
        results = manager.search(query, top_k=10, source_type=KGConfig.RELATION_SOURCE_TYPE)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Semantic relation search failed: {exc}")
        return
    if not results:
        st.info("No matching relations found.")
        return
    df = pd.DataFrame(
        [
            {
                "Entity 1": r.record.metadata.extra.get("subject", ""),
                "Relationship": r.record.metadata.extra.get("relation", ""),
                "Entity 2": r.record.metadata.extra.get("object", ""),
                "Relevance": round(r.score, 3),
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_similar_node_recommendation(
    manager: "EmbeddingManager", node_to_embedding_id: dict[str, str]
) -> None:
    """Show entities most similar (by embedding cosine similarity) to a chosen node."""
    if not node_to_embedding_id:
        st.info("No embedded entities available yet.")
        return
    selected_label = st.selectbox(
        "Select a node", sorted(node_to_embedding_id.keys()), key="kg_similar_node_select"
    )
    embedding_id = node_to_embedding_id.get(selected_label)
    if not embedding_id:
        return
    try:
        results = manager.find_similar(embedding_id, top_k=KGConfig.DEFAULT_SIMILAR_NODES_TOP_K)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Could not compute similar nodes: {exc}")
        return
    if not results:
        st.info("No similar nodes found.")
        return
    df = pd.DataFrame(
        [
            {"Similar Entity": r.record.metadata.title, "Similarity": round(r.score, 3)}
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_entity_clusters(
    manager: "EmbeddingManager", node_to_embedding_id: dict[str, str]
) -> None:
    """Automatically group semantically similar entities and display the clusters."""
    if not node_to_embedding_id:
        st.info("No embedded entities available yet.")
        return
    threshold = st.slider(
        "Cluster similarity threshold",
        0.50, 0.95, KGConfig.DEFAULT_CLUSTER_THRESHOLD, step=0.05,
        key="kg_cluster_threshold",
    )
    try:
        clusters = _cluster_entities_by_embedding(manager, node_to_embedding_id, threshold)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Entity clustering failed: %s", exc)
        st.error("Could not compute entity clusters.")
        return

    multi_clusters = [c for c in clusters if len(c) > 1]
    if not multi_clusters:
        st.info("No clusters of similar entities found at this threshold.")
        return
    for i, cluster in enumerate(sorted(multi_clusters, key=len, reverse=True), start=1):
        st.markdown(f"**Cluster {i}** ({len(cluster)} entities): " + ", ".join(cluster))


def _render_semantic_graph_expansion(
    manager: "EmbeddingManager",
    graph: "nx.DiGraph",
    node_to_embedding_id: dict[str, str],
) -> None:
    """Suggest additional, currently-absent nodes related to a selected node."""
    if not node_to_embedding_id:
        st.info("No embedded entities available yet.")
        return
    selected_label = st.selectbox(
        "Expand from node", sorted(node_to_embedding_id.keys()), key="kg_expand_node_select"
    )
    embedding_id = node_to_embedding_id.get(selected_label)
    if not embedding_id:
        return
    try:
        results = manager.find_similar(embedding_id, top_k=20)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Could not compute suggested connections: {exc}")
        return

    existing_labels = {str(n) for n in graph.nodes()}
    suggestions = [
        r for r in results if r.record.metadata.title not in existing_labels
    ][: KGConfig.DEFAULT_EXPANSION_TOP_K]

    if not suggestions:
        st.info("No new suggested connections found beyond the current graph.")
        return

    st.write(f"Suggested additional nodes related to **{selected_label}**:")
    for r in suggestions:
        st.markdown(f"- {r.record.metadata.title} (similarity: {r.score:.3f})")


def _render_semantic_question_support(
    manager: "EmbeddingManager", graph: "nx.DiGraph"
) -> None:
    """Answer natural-language questions by retrieving candidate entities via
    embeddings, then reading their existing graph connections."""
    question = st.text_input(
        "Ask a question about this graph "
        "(e.g. \"What proteins are related to breast cancer?\")",
        key="kg_question_input",
    )
    if not question:
        return
    try:
        entity_hits = manager.search(question, top_k=5, source_type=KGConfig.ENTITY_SOURCE_TYPE)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Question retrieval failed: {exc}")
        return
    if not entity_hits:
        st.info("No related entities found for this question.")
        return

    st.write("Candidate entities retrieved for this question:")
    for hit in entity_hits:
        label = hit.record.metadata.title
        st.markdown(f"**{label}** _(relevance: {hit.score:.3f})_")
        if label in graph:
            neighbors = sorted(
                {str(n) for n in list(graph.successors(label)) + list(graph.predecessors(label))}
            )
            if neighbors:
                st.caption("Connected to: " + ", ".join(neighbors[:10]))


def _render_related_paper_recommendations(
    manager: "EmbeddingManager", papers: list[Any]
) -> None:
    """If this graph originated from Literature Search results, recommend
    related papers using the embeddings already generated for those papers
    by the Literature Search module (reused here, not regenerated)."""
    if not papers:
        st.info("No source papers available for this graph.")
        return

    paper_options: dict[str, str] = {}
    for p in papers:
        title = getattr(p, "title", None) or "Untitled"
        identifier = (
            getattr(p, "id", None)
            or getattr(p, "doi", None)
            or getattr(p, "pmid", None)
        )
        if identifier:
            paper_options[title] = str(identifier)

    if not paper_options:
        st.info("No paper identifiers available for embedding-based recommendations.")
        return

    selected_title = st.selectbox(
        "Find papers related to", sorted(paper_options.keys()), key="kg_related_paper_select"
    )
    source_id = paper_options[selected_title]

    # Reuse: look up the embedding the Literature Search module already
    # generated for this paper; do not re-embed it here.
    existing = manager.get_by_source(source_id)
    if not existing:
        st.info(
            "This paper has not been embedded yet (Literature Search embeds "
            "papers on ingestion; re-run a search to generate it)."
        )
        return

    try:
        results = manager.find_similar(existing[0].embedding_id, top_k=6)
    except (EmbeddingManagerError, ValueError) as exc:
        st.error(f"Related paper search failed: {exc}")
        return
    if not results:
        st.info("No related papers found.")
        return
    for r in results:
        st.markdown(f"- **{r.record.metadata.title}** (similarity: {r.score:.3f})")


# ==============================================================================
# 5. VISUALIZATION (PyVis)
# ==============================================================================

def _render_legend() -> None:
    legend_html = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="color:{color}; font-weight:600;">●</span> {label}'
        for label, color in KGConfig.TYPE_LEGEND
    )
    st.markdown(legend_html, unsafe_allow_html=True)


@contextmanager
def _force_utf8_file_io():
    """Force any text-mode open() call that doesn't specify an encoding to
    use UTF-8 for the duration of the block.

    PyVis internally writes (and sometimes reads back) its rendered HTML via
    a plain open()/write() with no explicit encoding. On Windows this falls
    back to the OS default codepage (cp1252, reported as the "charmap"
    codec), which cannot represent many characters that show up in real
    biomedical abstracts — Greek letters (α, β, μ), en/em dashes, curly
    quotes, degree signs, etc. — and raises UnicodeEncodeError. Temporarily
    patching builtins.open to default to UTF-8 works around this without
    touching PyVis itself.
    """
    original_open = builtins.open

    def _utf8_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_open(file, mode, *args, **kwargs)

    builtins.open = _utf8_open
    try:
        yield
    finally:
        builtins.open = original_open


def _render_graph(
    graph: "nx.DiGraph",
    embedding_layout: Optional[dict[str, tuple[float, float]]] = None,
) -> None:
    """Render the filtered graph with PyVis.

    Parameters
    ----------
    embedding_layout:
        # --- EMBEDDINGS INTEGRATION ---
        Optional mapping of node label -> (x, y) produced by
        :func:`_compute_embedding_layout`. When provided, semantically
        similar nodes are given initial positions close to one another;
        PyVis's existing physics simulation still runs on top of this, so
        the existing force-directed layout behavior is fully preserved —
        embeddings only nudge the *starting* arrangement toward clusters of
        related concepts.
    """
    if not _HAS_PYVIS:
        st.error(
            "PyVis is not installed, so the interactive graph can't be rendered. "
            "Run `pip install pyvis` to enable this feature."
        )
        return

    if graph.number_of_nodes() == 0:
        st.warning(
            "No graph to display after filtering. Try lowering the minimum "
            "node importance or increasing the max nodes / max edges sliders."
        )
        return

    embedding_layout = embedding_layout or {}

    try:
        net = Network(
            height="800px",
            width="100%",
            directed=True,
            bgcolor="#0e1117",
            font_color="#f2f6f5",
            notebook=False,
            cdn_resources="in_line",
        )
        net.barnes_hut(
            gravity=-8000,
            central_gravity=0.3,
            spring_length=180,
            spring_strength=0.02,
            damping=0.9,
        )
        net.toggle_physics(True)

        for node, data in graph.nodes(data=True):
            ntype = _node_type(data)
            color = _node_color(ntype)
            importance = data.get("importance", 0.0)
            size = 16 + min(importance * 45, 34)

            node_kwargs = dict(
                label=str(node),
                title=f"{node} | type: {ntype} | importance: {importance:.2f}",
                color=color,
                size=size,
                font={"size": 16, "color": "#f2f6f5", "face": "arial"},
            )

            # --- EMBEDDINGS INTEGRATION ---
            # Seed the node's starting position from its embedding-based 2D
            # projection (if available) so semantically similar nodes start
            # out near one another. Physics remains on, so this only biases
            # the layout rather than replacing the existing behavior.
            position = embedding_layout.get(str(node))
            if position is not None:
                node_kwargs["x"], node_kwargs["y"] = position

            net.add_node(str(node), **node_kwargs)

        for u, v, data in graph.edges(data=True):
            rel = _edge_label(data)
            net.add_edge(
                str(u),
                str(v),
                label=rel,
                title=rel,
                color="#5c6773",
                arrows="to",
                font={"size": 11, "color": "#9aa5a1", "strokeWidth": 0},
            )

        net.set_options("""
        {
          "nodes": {"borderWidth": 1, "shadow": false},
          "edges": {
            "smooth": {"type": "dynamic"},
            "length": 220
          },
          "physics": {
            "barnesHut": {"avoidOverlap": 0.6},
            "stabilization": {"iterations": 200}
          },
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true}
        }
        """)

        # PyVis can still write/read an intermediate HTML file internally
        # even via generate_html(), using open() with no explicit encoding.
        # Force UTF-8 for that call so non-ASCII characters from real
        # abstracts (Greek letters, dashes, curly quotes, etc.) don't crash
        # on Windows' default cp1252 codepage.
        with _force_utf8_file_io():
            try:
                html_content = net.generate_html(notebook=False)
            except TypeError:
                # Older pyvis versions don't accept the `notebook` kwarg here.
                html_content = net.generate_html()

        st.components.v1.html(html_content, height=820, scrolling=True)

    except Exception as exc:  # noqa: BLE001
        logger.exception("PyVis rendering failed: %s", exc)
        st.error(f"Could not render the graph visualization: {exc}")


# ==============================================================================
# 6. SUMMARY + TABLE
# ==============================================================================

def _render_summary(papers_count: int, raw_entity_count: int, shown_nodes: int, shown_edges: int) -> None:
    st.subheader("Knowledge Graph Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers processed", papers_count)
    c2.metric("Entities detected", raw_entity_count)
    c3.metric("Important concepts shown", shown_nodes)
    c4.metric("Relationships shown", shown_edges)
    st.caption(
        f"To keep the graph human-readable, at most {KGConfig.TOP_NODES_CAP} concepts and "
        f"{KGConfig.TOP_EDGES_CAP} relationships are ever displayed, ranked by a blend of "
        f"connectivity (degree centrality) and how often they appear across your papers."
    )


def _render_relationship_table(graph: "nx.DiGraph") -> None:
    st.subheader("Important Biomedical Relationships")

    rows = [
        {"Entity 1": u, "Relationship": _edge_label(data), "Entity 2": v}
        for u, v, data in graph.edges(data=True)
    ]

    if not rows:
        st.info("No relationships survived filtering. Try adjusting the controls above.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ==============================================================================
# 7. MAIN ENTRY POINT
# ==============================================================================

def show() -> None:
    """Public entry point — call this from app.py to render the Knowledge
    Graph Explorer page."""
    st.title("🕸️ Knowledge Graph Explorer")
    st.caption(
        "Explore biomedical relationships (genes, diseases, drugs, proteins) "
        "extracted from your literature search results."
    )

    if not _HAS_NETWORKX:
        st.error("The `networkx` package is required for this module. Run `pip install networkx`.")
        return

    results = st.session_state.get("bx_last_results", [])
    papers_with_abstracts = [p for p in results if getattr(p, "abstract", "")]

    if not results or not papers_with_abstracts:
        st.info("Run literature search first to build a knowledge graph.")
        return

    if not _HAS_PIPELINE:
        st.error(
            "The Knowledge Graph pipeline (`knowledge_graph.kg_pipeline.KnowledgeGraphPipeline`) "
            "could not be imported. Make sure the `knowledge_graph` package and its dependencies "
            "(e.g. the scispaCy model `en_core_sci_sm`) are installed."
        )
        return

    st.markdown("### Graph Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_nodes = st.slider("Maximum nodes", 10, 100, KGConfig.DEFAULT_MAX_NODES, step=5)
    with c2:
        max_edges = st.slider("Maximum edges", 20, 200, KGConfig.DEFAULT_MAX_EDGES, step=10)
    with c3:
        min_importance = st.slider("Minimum node importance", 0.0, 1.0, 0.0, step=0.05)

    cache_key = (len(papers_with_abstracts), st.session_state.get("bx_last_query", ""))
    rebuild = st.button("🕸️ Build / Refresh Knowledge Graph", use_container_width=True)
    needs_build = rebuild or st.session_state.get("kg_cache_key") != cache_key

    if needs_build:
        with st.spinner(f"Extracting entities & relationships from {len(papers_with_abstracts)} papers…"):
            try:
                pipeline = KnowledgeGraphPipeline(entity_model=KGConfig.ENTITY_MODEL)
                process_result = pipeline.process_papers(papers_with_abstracts)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Knowledge graph pipeline failed: %s", exc)
                st.error(f"Knowledge graph construction failed: {exc}")
                return

            raw_graph = process_result.get("graph")
            st.session_state["kg_raw_graph"] = raw_graph
            st.session_state["kg_process_result"] = process_result
            st.session_state["kg_cache_key"] = cache_key

    raw_graph = st.session_state.get("kg_raw_graph")
    process_result = st.session_state.get("kg_process_result", {})

    if raw_graph is None:
        st.info("Click **Build / Refresh Knowledge Graph** to process your search results.")
        return

    if not isinstance(raw_graph, (nx.Graph, nx.DiGraph)):
        st.error("Unexpected graph format returned from the pipeline.")
        return

    if raw_graph.number_of_nodes() == 0:
        st.warning("No entities or relationships could be extracted from the current search results.")
        return

    failed = [d for d in process_result.get("details", []) if d.get("status") == "failed"]
    skipped = [d for d in process_result.get("details", []) if d.get("status") == "skipped"]
    if failed:
        with st.expander(f"⚠️ {len(failed)} paper(s) failed during processing"):
            for d in failed:
                st.write(f"- {d.get('error', 'Unknown error')}")
    if skipped:
        st.caption(f"{len(skipped)} paper(s) were skipped (no abstract).")

    try:
        filtered_graph, _importance = _build_filtered_graph(
            raw_graph, max_nodes=max_nodes, max_edges=max_edges, min_importance=min_importance
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph filtering failed: %s", exc)
        st.error(f"Could not filter the knowledge graph: {exc}")
        return

    _render_summary(
        papers_count=len(papers_with_abstracts),
        raw_entity_count=raw_graph.number_of_nodes(),
        shown_nodes=filtered_graph.number_of_nodes(),
        shown_edges=filtered_graph.number_of_edges(),
    )

    st.markdown("---")
    _render_relationship_table(filtered_graph)

    # --- EMBEDDINGS INTEGRATION ---------------------------------------------
    # Sync embeddings for the currently displayed entities/relations, reusing
    # anything already embedded (in this or a prior build) via EmbeddingManager,
    # and only generating (batched) embeddings for genuinely new items.
    # This does not alter graph construction/filtering above in any way.
    node_to_embedding_id: dict[str, str] = {}
    embedding_layout: dict[str, tuple[float, float]] = {}
    manager: Optional["EmbeddingManager"] = None

    if _HAS_EMBEDDINGS:
        manager = _get_embedding_manager()
        if manager is not None:
            try:
                with st.spinner("Syncing entity & relation embeddings…"):
                    node_to_embedding_id = _sync_entity_embeddings(filtered_graph, manager)
                    _sync_relation_embeddings(filtered_graph, manager)
                    embedding_layout = _compute_embedding_layout(manager, node_to_embedding_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Embedding sync failed: %s", exc)
                st.warning("Semantic features are temporarily unavailable (embedding sync failed).")
        else:
            st.warning("The embeddings module could not be initialized; semantic features are disabled.")
    else:
        st.caption(
            "Install the `embeddings` module's dependencies to enable semantic search, "
            "clustering, and recommendations for this graph."
        )
    # --- END EMBEDDINGS INTEGRATION (sync) ----------------------------------

    st.markdown("---")
    st.subheader("Knowledge Graph Visualization")
    _render_legend()
    # --- EMBEDDINGS INTEGRATION: pass the embedding-derived layout hint ----
    _render_graph(filtered_graph, embedding_layout=embedding_layout)
    # --- END EMBEDDINGS INTEGRATION -----------------------------------------

    # --- EMBEDDINGS INTEGRATION: semantic features section ------------------
    if _HAS_EMBEDDINGS and manager is not None:
        st.markdown("---")
        st.subheader("🧠 Semantic Intelligence")
        st.caption(
            "Powered by the existing `embeddings` module (EmbeddingManager) — "
            "reuses cached embeddings whenever possible."
        )
        tabs = st.tabs(
            [
                "Node Search",
                "Relation Search",
                "Similar Nodes",
                "Clusters",
                "Graph Expansion",
                "Ask a Question",
                "Related Papers",
            ]
        )
        with tabs[0]:
            _render_semantic_node_search(manager, node_to_embedding_id)
        with tabs[1]:
            _render_semantic_relation_search(manager)
        with tabs[2]:
            _render_similar_node_recommendation(manager, node_to_embedding_id)
        with tabs[3]:
            _render_entity_clusters(manager, node_to_embedding_id)
        with tabs[4]:
            _render_semantic_graph_expansion(manager, filtered_graph, node_to_embedding_id)
        with tabs[5]:
            _render_semantic_question_support(manager, filtered_graph)
        with tabs[6]:
            _render_related_paper_recommendations(manager, papers_with_abstracts)
    # --- END EMBEDDINGS INTEGRATION (semantic features section) -------------


# ==============================================================================
# Allow standalone execution for local testing: streamlit run knowledge_graphs.py
# ==============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="BioNexus — Knowledge Graph", page_icon="🕸️", layout="wide")
    show()