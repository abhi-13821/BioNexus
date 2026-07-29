"""
BioNexus Knowledge Graph Package
=================================

A modular, fault-tolerant pipeline for building and querying biomedical
knowledge graphs from unstructured text (e.g. PubMed abstracts).

This package is composed of four independent stages, each usable on its
own or chained together into a full pipeline:

    1. ``EntityExtractor``   -- Named Entity Recognition (spaCy / SciSpaCy)
    2. ``RelationExtractor`` -- Subject-Verb-Object relation extraction
    3. ``GraphBuilder``      -- Deduplicated, normalized graph construction
    4. ``QueryEngine``       -- Read-only querying, analytics, and export

Naming note
-----------
``entity_extractor.py`` and ``relation_extractor.py`` each define their
own local ``ModelLoadError`` for model-loading failures, kept separate
by design so those two modules remain fully independent of one another.
To avoid one silently shadowing the other in this shared namespace,
they are re-exported here as ``EntityModelLoadError`` and
``RelationModelLoadError`` respectively.

Example: full pipeline usage
-----------------------------
>>> from knowledge_graph import (
...     EntityExtractor,
...     RelationExtractor,
...     GraphBuilder,
...     QueryEngine,
... )
>>>
>>> text = "Aspirin inhibits COX-1 and reduces inflammation."
>>>
>>> # Stage 1: extract entities (non-strict so this runs even without a
>>> # model installed -- it will simply degrade to an empty result).
>>> entity_extractor = EntityExtractor(model_name="en_core_web_sm", strict=False)
>>> entities = entity_extractor.extract_entities(text)
>>>
>>> # Stage 2: extract relations.
>>> relation_extractor = RelationExtractor(model_name="en_core_web_sm", strict=False)
>>> relations = relation_extractor.extract_relations(text)
>>>
>>> # Stage 3: build the graph (deduplicates & normalizes automatically).
>>> builder = GraphBuilder()
>>> builder.build_from_extractions(entities, relations, source="pubmed:demo")
>>>
>>> # Stage 4: query the resulting graph.
>>> engine = QueryEngine(builder.get_graph())
>>> engine.get_targets("Aspirin")
>>> engine.graph_statistics()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------
from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from .graph_builder import GraphBuilder
from .query_engine import QueryEngine

# ---------------------------------------------------------------------------
# Public exceptions
#
# ModelLoadError is intentionally defined independently in both
# entity_extractor.py and relation_extractor.py (see module docstring
# above), so both are re-exported here under distinct, unambiguous names.
# ---------------------------------------------------------------------------
from .entity_extractor import (
    ModelLoadError as EntityModelLoadError,
    EntityExtractionError,
)
from .relation_extractor import (
    ModelLoadError as RelationModelLoadError,
    RelationExtractionError,
)
from .graph_builder import GraphBuildError
from .query_engine import QueryEngineError

__version__: str = "1.0.0"

__all__ = [
    # Core pipeline classes
    "EntityExtractor",
    "RelationExtractor",
    "GraphBuilder",
    "QueryEngine",
    # Exceptions
    "EntityModelLoadError",
    "EntityExtractionError",
    "RelationModelLoadError",
    "RelationExtractionError",
    "GraphBuildError",
    "QueryEngineError",
    # Metadata
    "__version__",
]