"""
BioNexus Real Literature -> Knowledge Graph Test
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frontend.literature_search import (
    SearchFilters,
    run_federated_search,
)

from knowledge_graph.kg_pipeline import KnowledgeGraphPipeline



def main():

    print("=" * 60)
    print("BioNexus Literature Knowledge Graph Pipeline")
    print("=" * 60)


    # -----------------------------
    # Literature Search
    # -----------------------------

    query = "EGFR mutation lung cancer"


    filters = SearchFilters(
        year_from=2015,
        year_to=2026,
        max_results=5,
        sort_by="Most Relevant",
    )


    sources = [
        "PubMed",
        "EuropePMC",
    ]


    print("\nSearching literature...")
    

    papers, status = run_federated_search(
        query,
        filters,
        sources
    )


    print("\nSearch Status:")
    print(status)


    print("\nPapers Found:")
    print(len(papers))


    if not papers:
        print("No papers found")
        return



    # -----------------------------
    # Knowledge Graph Pipeline
    # -----------------------------

    pipeline = KnowledgeGraphPipeline(
        entity_model="en_core_sci_sm"
    )


    print("\nBuilding Knowledge Graph...")


    result = pipeline.process_papers(
        papers
    )


    print("\nPipeline Result")
    print(result["papers_processed"])



    # -----------------------------
    # Graph Output
    # -----------------------------

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


    for u,v,data in graph.edges(data=True):

        print(
            f"{u} --{data['relation']}--> {v}"
        )


    print("\nReal Literature KG Test Completed Successfully")



if __name__ == "__main__":
    main()