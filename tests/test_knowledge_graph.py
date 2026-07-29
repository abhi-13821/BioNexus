"""
Test script for the BioNexus Knowledge Graph pipeline.
"""

from knowledge_graph.entity_extractor import EntityExtractor
from knowledge_graph.relation_extractor import RelationExtractor
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.query_engine import QueryEngine


def main():
    text = """
    Aspirin treats headache.
    Ibuprofen reduces inflammation.
    Paracetamol relieves fever.
    """

    print("=" * 60)
    print("INPUT TEXT")
    print("=" * 60)
    print(text)

    # -------------------------
    # Entity Extraction
    # -------------------------
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_entities(text)

    print("\nExtracted Entities:")
    for entity in entities:
        print(entity)

    # -------------------------
    # Relation Extraction
    # -------------------------
    relation_extractor = RelationExtractor()
    relations = relation_extractor.extract_relations(text)

    print("\nExtracted Relations:")
    for relation in relations:
        print(relation)

    # -------------------------
    # Graph Building
    # -------------------------
    graph = GraphBuilder()

    summary = graph.build_from_extractions(
        entities=entities,
        relations=relations,
        source="test_document",
    )

    print("\nGraph Summary")
    print(summary)

    nx_graph = graph.get_graph()

    print("\nNodes")
    for node, data in nx_graph.nodes(data=True):
        print(node, data)

    print("\nEdges")
    for u, v, data in nx_graph.edges(data=True):
        print(f"{u} --{data['relation']}--> {v}")

    # -------------------------
    # Query Engine
    # -------------------------
    qe = QueryEngine(nx_graph)

    print("\nTargets of aspirin")
    print(qe.get_targets("aspirin"))

    print("\nSources of headache")
    print(qe.get_sources("headache"))

    print("\nKnowledge Graph Test Completed Successfully")


if __name__ == "__main__":
    main()