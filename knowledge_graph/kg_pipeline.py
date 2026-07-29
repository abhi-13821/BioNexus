"""
kg_pipeline.py

BioNexus Literature → Knowledge Graph Integration Pipeline
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from knowledge_graph.entity_extractor import EntityExtractor
from knowledge_graph.relation_extractor import RelationExtractor
from knowledge_graph.graph_builder import GraphBuilder


logger = logging.getLogger(__name__)


class KnowledgeGraphPipeline:
    """
    Connects biomedical literature search results
    with the BioNexus Knowledge Graph.

    Flow:

    Paper abstracts
          |
          ↓
    Entity Extraction
          |
          ↓
    Relation Extraction
          |
          ↓
    Graph Construction
    """

    def __init__(
        self,
        entity_model: str = "en_core_sci_sm",
    ) -> None:

        self.entity_extractor = EntityExtractor(
            model_name=entity_model,
            strict=False
        )

        self.relation_extractor = RelationExtractor(
            model_name=entity_model,
            strict=False
        )

        self.graph_builder = GraphBuilder()


    def process_paper(
        self,
        paper: Any,
    ) -> dict:
        """
        Process a single research paper.

        Expected paper object:
        Paper.abstract
        Paper.pmid / doi / title
        """

        abstract = getattr(
            paper,
            "abstract",
            ""
        )

        if not abstract:
            return {
                "status": "skipped",
                "reason": "No abstract"
            }


        source = (
            getattr(paper, "pmid", None)
            or getattr(paper, "doi", None)
            or "unknown"
        )


        # -------------------------
        # Entity extraction
        # -------------------------

        entities = self.entity_extractor.extract_entities(
            abstract
        )


        # -------------------------
        # Relation extraction
        # -------------------------

        relations = self.relation_extractor.extract_relations(
            abstract
        )


        # -------------------------
        # Graph update
        # -------------------------

        summary = self.graph_builder.build_from_extractions(
            entities=entities,
            relations=relations,
            source=source
        )


        return {
            "status": "success",
            "entities": len(entities),
            "relations": len(relations),
            "summary": summary
        }



    def process_papers(
        self,
        papers: List[Any],
    ) -> dict:
        """
        Process multiple literature papers.
        """

        results = []

        for paper in papers:

            try:
                result = self.process_paper(
                    paper
                )

                results.append(result)

            except Exception as exc:

                logger.exception(
                    "Failed processing paper: %s",
                    exc
                )

                results.append(
                    {
                        "status": "failed",
                        "error": str(exc)
                    }
                )


        return {
            "papers_processed": len(results),
            "details": results,
            "graph": self.graph_builder.get_graph()
        }



    def get_graph(self):

        return self.graph_builder.get_graph()