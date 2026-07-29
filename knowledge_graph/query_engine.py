"""
query_engine.py
================

Read-only query and analysis layer for the BioNexus knowledge graph.

This module wraps a ``networkx.MultiDiGraph`` behind a single stable
interface (`QueryEngine`) that:

    * Never raises an uncaught / built-in exception to the caller.
    * Never crashes on missing entities, ``None`` input, or an empty
      graph.
    * Returns structured dictionaries (never raw tuples) for every
      relation-bearing result.
    * Supports graph export to GraphML and JSON with informative error
      handling for filesystem failures.

This file has **no dependency on any other module in this package** --
it only consumes a plain ``networkx.MultiDiGraph`` (e.g. the object
returned by ``GraphBuilder.get_graph()``), never the ``GraphBuilder``
class itself. This keeps it fully independent and testable in isolation.

Example
-------
>>> from query_engine import QueryEngine
>>> # graph = GraphBuilder(...).get_graph()  # or any MultiDiGraph
>>> engine = QueryEngine(graph)
>>> engine.get_targets("Aspirin")
>>> engine.shortest_path("Aspirin", "Inflammation")
>>> engine.graph_statistics()
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger (see entity_extractor.py for rationale).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions, scoped to this module only.
# ---------------------------------------------------------------------------
class QueryEngineError(Exception):
    """
    Raised for genuine, unrecoverable query-engine failures (e.g.
    NetworkX not installed, or an export operation that fails at the
    filesystem level). Missing entities/nodes do NOT raise this -- they
    are handled gracefully with empty/structured "not found" results.
    """


class QueryEngine:
    """
    Fault-tolerant read-only query interface over a knowledge graph.

    Attributes
    ----------
    graph : Any
        The underlying ``networkx.MultiDiGraph`` instance being queried.
    """

    def __init__(self, graph: Optional[Any] = None) -> None:
        """
        Initialize the query engine over a given graph.

        Parameters
        ----------
        graph : Optional[Any]
            A ``networkx.MultiDiGraph`` to query. If ``None``, an empty
            graph is created (useful for testing this class in
            isolation). If a non-``MultiDiGraph`` object is passed, a
            warning is logged and an empty graph is used instead.

        Raises
        ------
        QueryEngineError
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
            raise QueryEngineError(message) from exc

        self._nx = nx

        if graph is not None and isinstance(graph, nx.MultiDiGraph):
            self.graph: Any = graph
        else:
            if graph is not None:
                logger.warning(
                    "Provided graph is not a networkx.MultiDiGraph (got "
                    "%s). Initializing an empty graph instead.",
                    type(graph).__name__,
                )
            self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------
    def node_exists(self, entity: Any) -> bool:
        """
        Check whether an entity exists as a node in the graph.

        Parameters
        ----------
        entity : Any
            The entity name to check (will be normalized before lookup).

        Returns
        -------
        bool
            True if the (normalized) entity is present, False otherwise
            -- including when the input itself is invalid.
        """
        node_id = self._normalize_entity_name(entity)
        if node_id is None:
            return False
        return self.graph.has_node(node_id)

    def edge_exists(self, source: Any, target: Any) -> bool:
        """
        Check whether at least one relation exists from source to target.

        Parameters
        ----------
        source : Any
            The source entity name.
        target : Any
            The target entity name.

        Returns
        -------
        bool
            True if at least one directed edge exists between the
            (normalized) source and target, False otherwise.
        """
        source_id = self._normalize_entity_name(source)
        target_id = self._normalize_entity_name(target)
        if source_id is None or target_id is None:
            return False
        return self.graph.has_edge(source_id, target_id)

    # ------------------------------------------------------------------
    # Relation queries
    # ------------------------------------------------------------------
    def get_targets(self, entity: Any) -> List[Dict[str, Any]]:
        """
        Get all entities that ``entity`` has an outgoing relation to.

        Parameters
        ----------
        entity : Any
            The source entity name.

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured records, one per outgoing edge::

                {
                    "target": str,
                    "relation": str,
                    "confidence": float,
                    "source": str,
                    "timestamp": str,
                    "metadata": dict,
                }

            Returns ``[]`` if the entity does not exist or has no
            outgoing relations. Never raises on a missing entity.
        """
        node_id = self._normalize_entity_name(entity)
        if node_id is None or not self.graph.has_node(node_id):
            logger.warning("get_targets: entity %r not found in graph.", entity)
            return []

        records: List[Dict[str, Any]] = []
        try:
            for _, target_id, attrs in self.graph.out_edges(node_id, data=True):
                records.append(self._edge_attrs_to_record(target_id, attrs, "target"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_targets failed for %r: %s", entity, exc)
            return []
        return records

    def get_sources(self, entity: Any) -> List[Dict[str, Any]]:
        """
        Get all entities that have an outgoing relation to ``entity``.

        Parameters
        ----------
        entity : Any
            The target entity name.

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured records, one per incoming edge::

                {
                    "source": str,
                    "relation": str,
                    "confidence": float,
                    "source": str,
                    "timestamp": str,
                    "metadata": dict,
                }

            Returns ``[]`` if the entity does not exist or has no
            incoming relations. Never raises on a missing entity.
        """
        node_id = self._normalize_entity_name(entity)
        if node_id is None or not self.graph.has_node(node_id):
            logger.warning("get_sources: entity %r not found in graph.", entity)
            return []

        records: List[Dict[str, Any]] = []
        try:
            for source_id, _, attrs in self.graph.in_edges(node_id, data=True):
                records.append(self._edge_attrs_to_record(source_id, attrs, "source_entity"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_sources failed for %r: %s", entity, exc)
            return []
        return records

    def find_relation(self, source: Any, target: Any) -> List[Dict[str, Any]]:
        """
        Find all relations directed from ``source`` to ``target``.

        Because the graph is a ``MultiDiGraph``, multiple distinct
        relation types may exist between the same pair of entities
        (e.g. both "treats" and "studied_with").

        Parameters
        ----------
        source : Any
            The source entity name.
        target : Any
            The target entity name.

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured relation records (see ``get_targets``
            for the shape), or ``[]`` if no relation exists or either
            entity is missing.
        """
        source_id = self._normalize_entity_name(source)
        target_id = self._normalize_entity_name(target)
        if source_id is None or target_id is None:
            return []
        if not self.graph.has_edge(source_id, target_id):
            return []

        try:
            edge_data = self.graph.get_edge_data(source_id, target_id) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "find_relation failed for (%r, %r): %s", source, target, exc
            )
            return []

        records: List[Dict[str, Any]] = []
        for attrs in edge_data.values():
            if isinstance(attrs, dict):
                records.append(self._edge_attrs_to_record(target_id, attrs, "target"))
        return records

    # ------------------------------------------------------------------
    # Structural queries
    # ------------------------------------------------------------------
    def shortest_path(self, entity1: Any, entity2: Any) -> Dict[str, Any]:
        """
        Find the shortest directed path between two entities.

        Parameters
        ----------
        entity1 : Any
            The starting entity name.
        entity2 : Any
            The destination entity name.

        Returns
        -------
        Dict[str, Any]
            A structured result::

                {
                    "found": bool,
                    "path": List[str],   # normalized entity IDs, in order
                    "length": int,       # number of edges traversed
                }

            ``found`` is False (with an empty path and length 0) if
            either entity is missing or no path exists -- this method
            never raises ``NetworkXNoPath`` or ``NodeNotFound`` to the
            caller.
        """
        empty_result = {"found": False, "path": [], "length": 0}

        source_id = self._normalize_entity_name(entity1)
        target_id = self._normalize_entity_name(entity2)
        if source_id is None or target_id is None:
            return empty_result
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(
                "shortest_path: one or both entities not found (%r, %r).",
                entity1,
                entity2,
            )
            return empty_result

        try:
            path = self._nx.shortest_path(self.graph, source=source_id, target=target_id)
        except self._nx.NetworkXNoPath:
            logger.info("No path found between %r and %r.", entity1, entity2)
            return empty_result
        except self._nx.NodeNotFound:
            logger.warning(
                "shortest_path: node not found during traversal for (%r, %r).",
                entity1,
                entity2,
            )
            return empty_result
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "shortest_path failed unexpectedly for (%r, %r): %s",
                entity1,
                entity2,
                exc,
            )
            return empty_result

        return {"found": True, "path": path, "length": max(0, len(path) - 1)}

    def neighbors(self, entity: Any) -> Dict[str, List[str]]:
        """
        Get all directly connected neighbors of an entity, by direction.

        Parameters
        ----------
        entity : Any
            The entity name to inspect.

        Returns
        -------
        Dict[str, List[str]]
            ``{"predecessors": List[str], "successors": List[str]}``.
            Both lists are empty if the entity does not exist. Never
            raises on a missing entity.
        """
        node_id = self._normalize_entity_name(entity)
        if node_id is None or not self.graph.has_node(node_id):
            logger.warning("neighbors: entity %r not found in graph.", entity)
            return {"predecessors": [], "successors": []}

        try:
            predecessors = list(self.graph.predecessors(node_id))
            successors = list(self.graph.successors(node_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("neighbors failed for %r: %s", entity, exc)
            return {"predecessors": [], "successors": []}

        return {"predecessors": predecessors, "successors": successors}

    # ------------------------------------------------------------------
    # Graph-wide statistics
    # ------------------------------------------------------------------
    def graph_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics describing the current graph.

        Returns
        -------
        Dict[str, Any]
            A structured summary::

                {
                    "node_count": int,
                    "edge_count": int,
                    "density": float,
                    "is_weakly_connected": bool,
                    "weakly_connected_components": int,
                    "average_out_degree": float,
                    "top_degree_nodes": List[Dict[str, Any]],
                }

            All fields default to safe zero-values (rather than raising)
            if the graph is empty or a statistic cannot be computed.
        """
        stats: Dict[str, Any] = {
            "node_count": 0,
            "edge_count": 0,
            "density": 0.0,
            "is_weakly_connected": False,
            "weakly_connected_components": 0,
            "average_out_degree": 0.0,
            "top_degree_nodes": [],
        }

        try:
            stats["node_count"] = self.graph.number_of_nodes()
            stats["edge_count"] = self.graph.number_of_edges()
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph_statistics: failed to read node/edge counts: %s", exc)
            return stats

        if stats["node_count"] == 0:
            return stats

        try:
            stats["density"] = round(self._nx.density(self.graph), 4)
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph_statistics: failed to compute density: %s", exc)

        try:
            stats["is_weakly_connected"] = self._nx.is_weakly_connected(self.graph)
            stats["weakly_connected_components"] = self._nx.number_weakly_connected_components(
                self.graph
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph_statistics: failed to compute connectivity: %s", exc)

        try:
            stats["average_out_degree"] = round(
                stats["edge_count"] / stats["node_count"], 4
            )
        except ZeroDivisionError:
            stats["average_out_degree"] = 0.0

        try:
            degree_pairs = sorted(
                self.graph.degree(), key=lambda pair: pair[1], reverse=True
            )
            stats["top_degree_nodes"] = [
                {"entity": node_id, "degree": degree}
                for node_id, degree in degree_pairs[:5]
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("graph_statistics: failed to compute top-degree nodes: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # Export utilities
    # ------------------------------------------------------------------
    def export_graphml(self, filepath: str) -> bool:
        """
        Export the graph to a GraphML file.

        Note that GraphML requires scalar-compatible attribute values;
        any list/dict node or edge attributes (e.g. ``labels``,
        ``metadata``) are automatically JSON-stringified before export
        so the write never fails due to unsupported attribute types.

        Parameters
        ----------
        filepath : str
            Destination path for the ``.graphml`` file.

        Returns
        -------
        bool
            True if the export succeeded.

        Raises
        ------
        QueryEngineError
            If ``filepath`` is invalid, or the write fails due to a
            filesystem error (permissions, missing parent directory,
            disk full, etc.). The underlying ``OSError`` is never
            leaked directly to the caller.
        """
        if not isinstance(filepath, str) or not filepath.strip():
            raise QueryEngineError(
                "export_graphml requires a non-empty destination file path."
            )

        try:
            exportable_graph = self._make_graphml_safe_copy()
            self._nx.write_graphml(exportable_graph, filepath)
            logger.info("Exported graph to GraphML at '%s'.", filepath)
            return True
        except OSError as exc:
            raise QueryEngineError(
                f"Failed to write GraphML file at '{filepath}': {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QueryEngineError(
                f"Unexpected error exporting GraphML to '{filepath}': {exc}"
            ) from exc

    def _make_graphml_safe_copy(self) -> Any:
        """
        Build a copy of the graph with all non-scalar attributes
        JSON-stringified, since GraphML does not support list/dict
        attribute values.

        Returns
        -------
        Any
            A new ``MultiDiGraph`` safe to pass to ``nx.write_graphml``.
        """
        safe_graph = self.graph.copy()

        for _, attrs in safe_graph.nodes(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value)

        for _, _, attrs in safe_graph.edges(data=True):
            for key, value in list(attrs.items()):
                if isinstance(value, (list, dict)):
                    attrs[key] = json.dumps(value)

        return safe_graph

    def export_json(self, filepath: str) -> bool:
        """
        Export the graph to a JSON file using node-link format.

        Parameters
        ----------
        filepath : str
            Destination path for the ``.json`` file.

        Returns
        -------
        bool
            True if the export succeeded.

        Raises
        ------
        QueryEngineError
            If ``filepath`` is invalid, the graph cannot be serialized,
            or the write fails due to a filesystem error. The underlying
            ``OSError``/``TypeError`` is never leaked directly to the
            caller.
        """
        if not isinstance(filepath, str) or not filepath.strip():
            raise QueryEngineError(
                "export_json requires a non-empty destination file path."
            )

        try:
            data = self._nx.node_link_data(self.graph)
        except Exception as exc:  # noqa: BLE001
            raise QueryEngineError(
                f"Failed to serialize graph to node-link JSON structure: {exc}"
            ) from exc

        try:
            with open(filepath, "w", encoding="utf-8") as file_handle:
                json.dump(data, file_handle, indent=2, default=str)
            logger.info("Exported graph to JSON at '%s'.", filepath)
            return True
        except OSError as exc:
            raise QueryEngineError(
                f"Failed to write JSON file at '{filepath}': {exc}"
            ) from exc
        except TypeError as exc:
            raise QueryEngineError(
                f"Graph contains non-JSON-serializable data: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def get_node_metadata(self, entity: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve the full metadata dictionary stored on a node.

        Parameters
        ----------
        entity : Any
            The entity name to look up.

        Returns
        -------
        Optional[Dict[str, Any]]
            The node's attribute dictionary, or ``None`` if the entity
            does not exist. Never raises on a missing entity.
        """
        node_id = self._normalize_entity_name(entity)
        if node_id is None or not self.graph.has_node(node_id):
            return None
        try:
            return dict(self.graph.nodes[node_id])
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_node_metadata failed for %r: %s", entity, exc)
            return None

    @staticmethod
    def _edge_attrs_to_record(
        other_node_id: str, attrs: Dict[str, Any], other_key_name: str
    ) -> Dict[str, Any]:
        """
        Convert raw edge attributes into a structured query result record.

        Parameters
        ----------
        other_node_id : str
            The node ID at the "other end" of this edge relative to the
            entity that was queried (i.e. the target for outgoing edges,
            the source for incoming edges).
        attrs : Dict[str, Any]
            The raw edge attribute dictionary as stored in the graph.
        other_key_name : str
            The dictionary key under which ``other_node_id`` should be
            reported (e.g. ``"target"`` or ``"source_entity"``).

        Returns
        -------
        Dict[str, Any]
            A structured, defensively-defaulted relation record.
        """
        return {
            other_key_name: other_node_id,
            "relation": attrs.get("relation", ""),
            "confidence": attrs.get("confidence", 0.0),
            "source": attrs.get("source", ""),
            "timestamp": attrs.get("timestamp", ""),
            "metadata": attrs.get("metadata", {}),
        }

    @staticmethod
    def _normalize_entity_name(name: Any) -> Optional[str]:
        """
        Normalize an entity name into the same form used as a node ID
        by ``graph_builder.py`` (lower-cased, whitespace-collapsed,
        punctuation preserved).

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

    def __repr__(self) -> str:
        """Return a concise, debug-friendly representation."""
        try:
            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()
        except Exception:  # noqa: BLE001
            node_count, edge_count = "?", "?"
        return f"QueryEngine(nodes={node_count}, edges={edge_count})"


# ---------------------------------------------------------------------------
# Example usage (only runs when this file is executed directly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import networkx as nx

    logging.basicConfig(level=logging.INFO)

    demo_graph = nx.MultiDiGraph()
    demo_graph.add_node("aspirin", canonical_name="Aspirin", labels=["CHEMICAL"])
    demo_graph.add_node("cox-1", canonical_name="COX-1", labels=["GENE"])
    demo_graph.add_node("inflammation", canonical_name="Inflammation", labels=["DISEASE"])
    demo_graph.add_edge(
        "aspirin", "cox-1", relation="inhibits", confidence=0.9,
        source="demo", timestamp="2026-01-01T00:00:00+00:00", metadata={},
    )
    demo_graph.add_edge(
        "cox-1", "inflammation", relation="mediates", confidence=0.7,
        source="demo", timestamp="2026-01-01T00:00:00+00:00", metadata={},
    )

    engine = QueryEngine(demo_graph)

    print(f"Engine status: {engine!r}")
    print("Targets of 'Aspirin':", engine.get_targets("Aspirin"))
    print("Sources of 'COX-1':", engine.get_sources("COX-1"))
    print("Path Aspirin -> Inflammation:", engine.shortest_path("Aspirin", "Inflammation"))
    print("Neighbors of 'COX-1':", engine.neighbors("COX-1"))
    print("Statistics:", engine.graph_statistics())