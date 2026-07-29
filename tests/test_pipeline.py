"""
End-to-end test:
Literature paper -> Knowledge Graph
"""

from knowledge_graph.kg_pipeline import KnowledgeGraphPipeline


class DummyPaper:
    """
    Simulates literature_search.Paper
    """

    title = "Aspirin treatment in inflammation"

    pmid = "TEST001"

    abstract = """
    Aspirin reduces inflammation.
    Ibuprofen treats pain.
    Paracetamol reduces fever.
    """



def main():

    print("=" * 60)
    print("BioNexus Knowledge Graph Pipeline Test")
    print("=" * 60)


    pipeline = KnowledgeGraphPipeline(
        entity_model="en_core_sci_sm"
    )


    paper = DummyPaper()


    result = pipeline.process_paper(
        paper
    )


    print("\nProcessing Result:")
    print(result)


    graph = pipeline.get_graph()


    print("\nGraph Statistics")
    print("----------------")
    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )


    print("\nRelations")
    print("----------------")

    for u, v, data in graph.edges(data=True):

        print(
            f"{u} --{data['relation']}--> {v}"
        )


    print("\nPipeline Test Completed Successfully")


if __name__ == "__main__":
    main()