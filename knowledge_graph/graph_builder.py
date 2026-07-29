"""
graph_builder.py
=================

Knowledge graph construction and maintenance layer for the BioNexus
pipeline.

This module wraps a ``networkx.MultiDiGraph`` behind a single stable
interface (`GraphBuilder`) that:

    * Never raises an uncaught / built-in exception to the caller.
    * Never crashes on missing files, ``None`` values, or malformed
      entity/relation dictionaries.
    * Automatically normalizes entity names, removes duplicate nodes,
      and merges duplicate relations into existing edges.
    * Stores rich, structured metadata on every node and edge.

This file has **no dependency on any other module in this package** --
it only consumes plain dictionaries shaped like the output of
``entity_extractor.py`` / ``relation_extractor.py``, never the classes
themselves. This keeps it fully independent and testable in isolation,
and keeps the extraction layer decoupled from the storage layer.

Example
-------
>>> from graph_builder import GraphBuilder
>>> builder = GraphBuilder()
>>> builder.add_entity({"text": "Aspirin", "label": "CHEMICAL"})
>>> builder.add_entity({"text": "COX-1", "label": "GENE"})
>>> builder.add_relation(
...     {"subject": "Aspirin", "verb": "inhibits", "object": "COX-1",
...      "confidence": 0.9, "sentence": "Aspirin inhibits COX-1."},
...     source="pubmed:12345",
... )
>>> graph = builder.get_graph()
>>> graph.number_of_nodes()
2
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Module-level logger (see entity_extractor.py for rationale).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions, scoped to this module only.
# ---------------------------------------------------------------------------
class GraphBuildError(Exception):
    """
    Raised for genuine, unrecoverable graph-construction failures (e.g.
    NetworkX is not installed). Malformed individual records do NOT
    raise this -- they are skipped and logged instead.
    """


# Default value used whenever an entity's type/label cannot be determined.
_UNKNOWN_LABEL = "UNKNOWN"

# Default value used whenever a relation's source is not specified.
_UNKNOWN_SOURCE = "unknown"

# Default confidence assigned to a relation that does not specify one.
_DEFAULT_CONFIDENCE = 0.5


class GraphBuilder:
    """
    Fault-tolerant builder and maintainer of a biomedical knowledge graph.

    Internally backed by a ``networkx.MultiDiGraph``, which allows
    multiple distinct relation types between the same pair of entities
    (e.g. "Aspirin -inhibits-> COX-1" and "Aspirin -treats-> Headache"
    both existing independently) while still merging genuinely duplicate
    relations together.

    Attributes
    ----------
    graph : Any
        The underlying ``networkx.MultiDiGraph`` instance. Exposed
        directly for advanced use cases (e.g. custom NetworkX
        algorithms), but callers are encouraged to use this class's
        methods for all mutation to preserve normalization/merging
        guarantees.
    """

    def __init__(self, existing_graph: Optional[Any] = None) -> None:
        """
        Initialize the graph builder.

        Parameters
        ----------
        existing_graph : Optional[Any]
            An existing ``networkx.MultiDiGraph`` to continue building
            on. If ``None`` (default), a fresh empty graph is created.
            If provided but not a ``MultiDiGraph``, a warning is logged
            and a new empty graph is created instead (the invalid graph
            is never mutated or discarded silently without notice).

        Raises
        ------
        GraphBuildError
            If NetworkX is not installed in this environment.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            message = (
                "networkx is not installed in this environment. Install "
                "it with:  pip install networkx"
            )
            logger.error("%s (original error: %s)", message, exc)
            raise GraphBuildError(message) from exc

        self._nx = nx  # Kept as an instance attribute, not a global.

        if existing_graph is not None:
            if isinstance(existing_graph, nx.MultiDiGraph):
                self.graph: Any = existing_graph
            else:
                logger.warning(
                    "existing_graph was provided but is not a "
                    "networkx.MultiDiGraph (got %s). Creating a new empty "
                    "graph instead.",
                    type(existing_graph).__name__,
                )
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Entity (node) management
    # ------------------------------------------------------------------
    def add_entity(
        self,
        entity: Union[str, Dict[str, Any]],
        source: str = _UNKNOWN_SOURCE,
    ) -> Optional[str]:
        """
        Add an entity to the graph, or merge it into an existing node.

        Parameters
        ----------
        entity : Union[str, Dict[str, Any]]
            Either a plain entity name, or a structured entity
            dictionary shaped like ``entity_extractor.py``'s output
            (must contain at least a ``"text"`` key if a dict).
        source : str, optional
            An identifier for where this entity was observed (e.g. a
            PubMed ID). Defaults to ``"unknown"``.

        Returns
        -------
        Optional[str]
            The normalized node ID used in the graph, or ``None`` if the
            entity could not be added (e.g. empty/invalid name). This
            method never raises on malformed input.
        """
        raw_text, label = self._coerce_entity_input(entity)
        node_id = self._normalize_entity_name(raw_text)

        if node_id is None:
            logger.warning(
                "Skipping entity with invalid or empty name: %r", entity
            )
            return None

        safe_source = source if isinstance(source, str) and source else _UNKNOWN_SOURCE
        now = self._current_timestamp()

        if self.graph.has_node(node_id):
            self._merge_entity_into_node(node_id, raw_text, label, safe_source, now)
        else:
            self.graph.add_node(
                node_id,
                canonical_name=raw_text,
                labels=[label],
                mention_count=1,
                first_seen=now,
                last_seen=now,
                sources=[safe_source],
            )
            logger.debug("Added new node '%s'.", node_id)

        return node_id

    def _merge_entity_into_node(
        self,
        node_id: str,
        raw_text: str,
        label: str,
        source: str,
        timestamp: str,
    ) -> None:
        """
        Merge a newly observed mention of an entity into its existing node.

        Parameters
        ----------
        node_id : str
            The normalized node identifier already present in the graph.
        raw_text : str
            The surface form of this particular mention.
        label : str
            The entity type/label of this particular mention.
        source : str
            Where this mention was observed.
        timestamp : str
            ISO-8601 timestamp of this observation.
        """
        try:
            attrs = self.graph.nodes[node_id]
            attrs["mention_count"] = attrs.get("mention_count", 0) + 1
            attrs["last_seen"] = timestamp

            labels = attrs.get("labels", [])
            if label not in labels:
                labels.append(label)
            attrs["labels"] = labels

            sources = attrs.get("sources", [])
            if source not in sources:
                sources.append(source)
            attrs["sources"] = sources

            logger.debug(
                "Merged mention of '%s' into existing node '%s'.",
                raw_text,
                node_id,
            )
        except KeyError as exc:
            # Should not happen given has_node() was already checked, but
            # guarded defensively per the "avoid KeyError" requirement.
            logger.warning(
                "Node '%s' disappeared during merge; skipping merge: %s",
                node_id,
                exc,
            )

    @staticmethod
    def _coerce_entity_input(entity: Union[str, Dict[str, Any]]) -> "tuple[str, str]":
        """
        Normalize the two accepted entity input shapes into ``(text, label)``.

        Parameters
        ----------
        entity : Union[str, Dict[str, Any]]
            Either a bare string or an entity dictionary.

        Returns
        -------
        tuple[str, str]
            A ``(text, label)`` pair. Both default to empty-string /
            "UNKNOWN" respectively if extraction fails for any reason.
        """
        if isinstance(entity, str):
            return entity, _UNKNOWN_LABEL
        if isinstance(entity, dict):
            text = entity.get("text", "")
            label = entity.get("label", _UNKNOWN_LABEL)
            text = text if isinstance(text, str) else ""
            label = label if isinstance(label, str) and label else _UNKNOWN_LABEL
            return text, label
        return "", _UNKNOWN_LABEL

    # ------------------------------------------------------------------
    # Relation (edge) management
    # ------------------------------------------------------------------
    def add_relation(
        self,
        relation: Dict[str, Any],
        source: str = _UNKNOWN_SOURCE,
        timestamp: Optional[str] = None,
    ) -> Optional[int]:
        """
        Add a relation to the graph, or merge it into an existing edge.

        Both the subject and object are automatically added as nodes
        (via ``add_entity``) if they do not already exist, so this
        method can be called directly on raw relation-extractor output
        without a separate entity-registration pass.

        Parameters
        ----------
        relation : Dict[str, Any]
            A relation dictionary shaped like ``relation_extractor.py``'s
            output. Must contain non-empty ``"subject"``, ``"verb"``,
            and ``"object"`` string values to be added; anything else is
            skipped gracefully.
        source : str, optional
            An identifier for where this relation was observed (e.g. a
            PubMed ID). Defaults to ``"unknown"``.
        timestamp : Optional[str]
            ISO-8601 timestamp to record for this relation. If ``None``,
            the current UTC time is used.

        Returns
        -------
        Optional[int]
            The MultiDiGraph edge key of the created or merged edge, or
            ``None`` if the relation could not be added. This method
            never raises on malformed input.
        """
        if not isinstance(relation, dict):
            logger.warning(
                "add_relation expected a dict, got %s; skipping.",
                type(relation).__name__,
            )
            return None

        subject_raw = relation.get("subject", "")
        verb_raw = relation.get("verb", "")
        object_raw = relation.get("object", "")
        confidence = relation.get("confidence", _DEFAULT_CONFIDENCE)
        sentence = relation.get("sentence", "")
        extra_metadata = relation.get("metadata", {})

        if not self._is_nonempty_str(subject_raw) or not self._is_nonempty_str(
            object_raw
        ) or not self._is_nonempty_str(verb_raw):
            logger.warning(
                "Skipping relation with missing subject/verb/object: %r",
                relation,
            )
            return None

        confidence = self._coerce_confidence(confidence)
        safe_source = source if isinstance(source, str) and source else _UNKNOWN_SOURCE
        safe_timestamp = timestamp if isinstance(timestamp, str) and timestamp else (
            self._current_timestamp()
        )
        extra_metadata = extra_metadata if isinstance(extra_metadata, dict) else {}

        subject_id = self.add_entity(subject_raw, source=safe_source)
        object_id = self.add_entity(object_raw, source=safe_source)

        if subject_id is None or object_id is None:
            logger.warning(
                "Could not normalize subject/object for relation: %r", relation
            )
            return None

        relation_label = self._normalize_relation_label(verb_raw)

        existing_key = self._find_matching_edge_key(
            subject_id, object_id, relation_label
        )

        if existing_key is not None:
            self._merge_relation_into_edge(
                subject_id,
                object_id,
                existing_key,
                confidence,
                safe_source,
                safe_timestamp,
            )
            return existing_key

        edge_key = self.graph.add_edge(
            subject_id,
            object_id,
            relation=relation_label,
            confidence=confidence,
            source=safe_source,
            timestamp=safe_timestamp,
            sentence=sentence if isinstance(sentence, str) else "",
            metadata={
                **extra_metadata,
                "merge_count": 1,
                "all_sources": [safe_source],
            },
        )
        logger.debug(
            "Added new edge '%s' -[%s]-> '%s'.",
            subject_id,
            relation_label,
            object_id,
        )
        return edge_key

    def _find_matching_edge_key(
        self, subject_id: str, object_id: str, relation_label: str
    ) -> Optional[int]:
        """
        Search existing parallel edges between two nodes for a duplicate.

        Parameters
        ----------
        subject_id : str
            Normalized source node ID.
        object_id : str
            Normalized target node ID.
        relation_label : str
            Normalized relation label to match against.

        Returns
        -------
        Optional[int]
            The key of a matching existing edge, or ``None`` if no
            duplicate relation was found.
        """
        try:
            edge_data = self.graph.get_edge_data(subject_id, object_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to inspect existing edges: %s", exc)
            return None

        if not edge_data:
            return None

        for key, attrs in edge_data.items():
            if isinstance(attrs, dict) and attrs.get("relation") == relation_label:
                return key
        return None

    def _merge_relation_into_edge(
        self,
        subject_id: str,
        object_id: str,
        edge_key: int,
        confidence: float,
        source: str,
        timestamp: str,
    ) -> None:
        """
        Merge a newly observed duplicate relation into an existing edge.

        The merge strategy keeps the higher of the two confidence scores
        (optimistic merge), records the newest timestamp, and appends
        the new source to the edge's provenance list without discarding
        the original ``source`` field's contract (req: edges store a
        single ``source`` field reflecting the most recent observation).

        Parameters
        ----------
        subject_id : str
            Normalized source node ID.
        object_id : str
            Normalized target node ID.
        edge_key : int
            The key of the edge to merge into.
        confidence : float
            The confidence score of the new (duplicate) observation.
        source : str
            The source of the new observation.
        timestamp : str
            ISO-8601 timestamp of the new observation.
        """
        try:
            attrs = self.graph.edges[subject_id, object_id, edge_key]
        except Exception as exc:  # noqa: BLE001 - covers KeyError and others
            logger.warning(
                "Edge (%s, %s, %s) disappeared during merge; skipping: %s",
                subject_id,
                object_id,
                edge_key,
                exc,
            )
            return

        attrs["confidence"] = max(attrs.get("confidence", 0.0), confidence)
        attrs["timestamp"] = timestamp
        attrs["source"] = source

        metadata = attrs.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["merge_count"] = metadata.get("merge_count", 1) + 1
        all_sources = metadata.get("all_sources", [])
        if source not in all_sources:
            all_sources.append(source)
        metadata["all_sources"] = all_sources
        attrs["metadata"] = metadata

        logger.debug(
            "Merged duplicate relation into edge (%s, %s, key=%s).",
            subject_id,
            object_id,
            edge_key,
        )

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------
    def build_from_extractions(
        self,
        entities: Optional[List[Dict[str, Any]]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        source: str = _UNKNOWN_SOURCE,
    ) -> Dict[str, int]:
        """
        Convenience method to bulk-load entities and relations at once.

        Parameters
        ----------
        entities : Optional[List[Dict[str, Any]]]
            A list of entity dictionaries (e.g. straight from
            ``EntityExtractor.extract_entities``). May be ``None`` or
            empty.
        relations : Optional[List[Dict[str, Any]]]
            A list of relation dictionaries (e.g. straight from
            ``RelationExtractor.extract_relations``). May be ``None`` or
            empty.
        source : str, optional
            An identifier applied to every entity/relation in this
            batch that doesn't already carry its own source.

        Returns
        -------
        Dict[str, int]
            A small summary: ``{"entities_added": int, "relations_added": int,
            "entities_skipped": int, "relations_skipped": int}``. Bad
            individual records are skipped, never raised.
        """
        summary = {
            "entities_added": 0,
            "entities_skipped": 0,
            "relations_added": 0,
            "relations_skipped": 0,
        }

        for entity in entities or []:
            result = self.add_entity(entity, source=source)
            if result is not None:
                summary["entities_added"] += 1
            else:
                summary["entities_skipped"] += 1

        for relation in relations or []:
            result = self.add_relation(relation, source=source)
            if result is not None:
                summary["relations_added"] += 1
            else:
                summary["relations_skipped"] += 1

        logger.info("build_from_extractions summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Removal / maintenance utilities
    # ------------------------------------------------------------------
    def remove_entity(self, entity_name: str) -> bool:
        """
        Remove an entity (and all its incident edges) from the graph.

        Parameters
        ----------
        entity_name : str
            The entity name to remove (will be normalized before lookup).

        Returns
        -------
        bool
            True if a node was found and removed, False otherwise.
        """
        node_id = self._normalize_entity_name(entity_name)
        if node_id is None or not self.graph.has_node(node_id):
            return False
        self.graph.remove_node(node_id)
        logger.debug("Removed node '%s'.", node_id)
        return True

    def clear(self) -> None:
        """Remove all nodes and edges, resetting the graph to empty."""
        self.graph.clear()
        logger.info("Graph cleared.")

    def get_graph(self) -> Any:
        """
        Return the underlying ``networkx.MultiDiGraph`` instance.

        Returns
        -------
        Any
            The live graph object. Mutating it directly bypasses this
            class's normalization/merging guarantees -- prefer
            ``add_entity`` / ``add_relation`` where possible.
        """
        return self.graph

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_entity_name(name: Any) -> Optional[str]:
        """
        Normalize an entity name into a stable node identifier.

        Normalization lower-cases the text and collapses internal
        whitespace, but deliberately preserves internal punctuation
        (e.g. hyphens in "COX-1") since it can be semantically
        meaningful in biomedical terminology.

        Parameters
        ----------
        name : Any
            The raw candidate entity name.

        Returns
        -------
        Optional[str]
            The normalized name, or ``None`` if the input was not a
            usable non-empty string.
        """
        if not isinstance(name, str):
            return None
        collapsed = re.sub(r"\s+", " ", name).strip().lower()
        return collapsed if collapsed else None

    @staticmethod
    def _normalize_relation_label(verb: Any) -> str:
        """
        Normalize a relation/verb label for consistent edge comparison.

        Parameters
        ----------
        verb : Any
            The raw candidate relation label.

        Returns
        -------
        str
            The normalized label, or an empty string if unusable.
        """
        if not isinstance(verb, str):
            return ""
        return re.sub(r"\s+", " ", verb).strip().lower()

    @staticmethod
    def _is_nonempty_str(value: Any) -> bool:
        """
        Check whether a value is a non-empty, non-whitespace-only string.

        Parameters
        ----------
        value : Any
            The value to check.

        Returns
        -------
        bool
            True if ``value`` is a string with visible content.
        """
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        """
        Safely coerce a confidence value into a float within [0.0, 1.0].

        Parameters
        ----------
        value : Any
            The raw confidence value, of any type.

        Returns
        -------
        float
            A valid confidence score. Falls back to the module default
            if the input cannot be interpreted as a number.
        """
        try:
            score = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid confidence value %r; defaulting to %.2f.",
                value,
                _DEFAULT_CONFIDENCE,
            )
            return _DEFAULT_CONFIDENCE
        return max(0.0, min(1.0, score))

    @staticmethod
    def _current_timestamp() -> str:
        """
        Get the current UTC time as an ISO-8601 string.

        Returns
        -------
        str
            The current timestamp, e.g. "2026-07-11T12:00:00+00:00".
        """
        return datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation."""
        try:
            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()
        except Exception:  # noqa: BLE001
            node_count, edge_count = "?", "?"
        return f"GraphBuilder(nodes={node_count}, edges={edge_count})"


# ---------------------------------------------------------------------------
# Example usage (only runs when this file is executed directly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    builder = GraphBuilder()

    sample_entities = [
        {"text": "Aspirin", "label": "CHEMICAL"},
        {"text": "  aspirin ", "label": "CHEMICAL"},  # Duplicate, different casing/whitespace.
        {"text": "COX-1", "label": "GENE"},
    ]
    sample_relations = [
        {
            "subject": "Aspirin",
            "verb": "inhibit",  # relation_extractor.py emits lemmatized verbs,
            "object": "COX-1",   # so both mentions normalize to "inhibit".
            "confidence": 0.8,
            "sentence": "Aspirin inhibits COX-1.",
        },
        {
            "subject": "  Aspirin ",  # Same relation, different casing/whitespace.
            "verb": "INHIBIT",
            "object": "cox-1",
            "confidence": 0.95,
            "sentence": "Studies confirm aspirin strongly inhibits COX-1.",
        },
    ]

    result_summary = builder.build_from_extractions(
        entities=sample_entities,
        relations=sample_relations,
        source="demo",
    )

    print(f"Builder status: {builder!r}")
    print(f"Build summary: {result_summary}")