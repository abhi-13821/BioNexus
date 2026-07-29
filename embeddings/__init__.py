"""
BioNexus Embeddings Module
===========================

This package provides the vector embedding infrastructure for the BioNexus
biomedical AI platform. It is responsible for converting biomedical text
(paper titles, abstracts, and full-text segments) into dense vector
representations, persisting those vectors alongside their metadata, and
enabling semantic similarity search over them.

The embeddings module is the foundational layer for:
    - Semantic search over biomedical literature.
    - A future FAISS-backed vector database.
    - Retrieval-Augmented Generation (RAG) pipelines.
    - Future Multi-Agent AI components that require contextual retrieval.

Package Structure
------------------
embeddings/
├── __init__.py           - Package initialization and public API surface.
├── embedding_generator.py - Generates vector embeddings from text.
├── embedding_store.py     - Persists embeddings and associated metadata.
├── embedding_search.py    - Performs semantic similarity search.
└── models.py              - Data models/schemas shared across the module.

Design Principles
------------------
- Single Responsibility: each module handles one concern (generation,
  storage, search, or data modeling).
- Dependency Inversion: higher-level components (e.g., search) depend on
  abstractions/interfaces exposed by lower-level components, not concrete
  implementations, so the embedding backend or storage backend can be
  swapped without breaking consumers.
- Explicit contracts: all public classes and functions are fully typed
  and documented so the Literature Search and Knowledge Graph modules
  (and future RAG/Multi-Agent modules) can integrate predictably.

Logging
-------
This package attaches a ``NullHandler`` to its root logger, following
Python library best practices. This ensures BioNexus's embeddings module
never emits log output by default and defers all logging configuration
(handlers, formatters, levels) to the consuming application (``app.py``).

Versioning
----------
The module exposes a local ``__version__`` string to allow other parts of
BioNexus (or diagnostics/tests) to introspect the embeddings module version
independently of the overall project version.

Note
----
As additional modules (``embedding_generator.py``, ``embedding_store.py``,
``embedding_search.py``, ``models.py``) are generated in subsequent steps,
their public classes/functions will be explicitly re-exported here via
``__all__`` to form a clean public API, e.g.:

    from embeddings import EmbeddingGenerator, EmbeddingStore, EmbeddingSearchEngine

This file will be revisited (not "modified" in the sense of altering
completed functionality, but "completed") once those modules exist, since
an ``__init__.py`` for a multi-module package is inherently built up
incrementally as the package grows.
"""

from __future__ import annotations

import logging

__version__: str = "0.1.0"

# Attach a NullHandler to this package's logger so that importing
# BioNexus's embeddings module never produces log output on its own.
# The consuming application (app.py) is responsible for configuring
# logging handlers, formatters, and levels.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__: list[str] = [
    "__version__",
]