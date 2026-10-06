"""
Simplified BioNexus Benchmark
"""
import sys
sys.path.insert(0, 'C:/projects/BioNexus')

import asyncio
import time
from datetime import datetime

from agents.literature_agent import LiteratureAgent
from agents.drug_agent import DrugAgent
from agents.smiles_agent import SmilesAgent
from agents.models import AgentRequest

# Test data
LITERATURE_QUERIES = [
    "EGFR mutations in lung cancer",
    "BRCA1 breast cancer treatment",
    "COVID-19 vaccine efficacy",
    "Alzheimer's disease biomarkers",
    "CRISPR gene editing",
]

DRUGS = ["Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Atorvastatin"]

SMILES_LIST = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C1=CC=CC=C1",  # Benzene
]


async def test_literature():
    print("\n📚 TESTING LITERATURE AGENT")
    print("-" * 40)
    agent = LiteratureAgent()
    await agent.start()
    
    results = []
    for query in LITERATURE_QUERIES:
        start = time.time()
        req = AgentRequest(instruction=query, parameters={"query": query})
        resp = await agent.handle_request(req)
        elapsed = (time.time() - start) * 1000
        
        papers = 0
        if resp.results and resp.results[0].output:
            papers = len(resp.results[0].output.get("papers", []))
        
        results.append({"query": query, "papers": papers, "latency_ms": elapsed})
        status = "✅" if papers > 0 else "❌"
        print(f"  {status} {query[:35]}... -> {papers} papers ({elapsed:.0f}ms)")
    
    await agent.stop()
    return results


async def test_drug():
    print("\n💊 TESTING DRUG AGENT")
    print("-" * 40)
    agent = DrugAgent()
    await agent.start()
    
    results = []
    for drug in DRUGS:
        req = AgentRequest(instruction=drug, parameters={"drug_name": drug})
        resp = await agent.handle_request(req)
        
        found = False
        if resp.results and resp.results[0].output:
            compound = resp.results[0].output.get("compound", {})
            found = bool(compound)
        
        results.append({"drug": drug, "found": found})
        print(f"  {'✅' if found else '❌'} {drug}")
    
    await agent.stop()
    return results


async def test_smiles():
    print("\n🧪 TESTING SMILES AGENT")
    print("-" * 40)
    agent = SmilesAgent()
    await agent.start()
    
    results = []
    for smiles in SMILES_LIST:
        req = AgentRequest(instruction=smiles, parameters={"smiles": smiles})
        resp = await agent.handle_request(req)
        
        props = {}
        if resp.results and resp.results[0].output:
            props = resp.results[0].output.get("properties", {})
        
        results.append({"smiles": smiles[:30], "props": len(props)})
        status = "✅" if props else "❌"
        print(f"  {status} {smiles[:30]}... -> {len(props)} properties")
    
    await agent.stop()
    return results


async def main():
    print("=" * 60)
    print("🚀 BIOENEXUS BENCHMARK")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    lit_results = await test_literature()
    drug_results = await test_drug()
    smiles_results = await test_smiles()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    lit_success = sum(1 for r in lit_results if r["papers"] > 0)
    print(f"\n📚 Literature Agent:")
    print(f"  Success: {lit_success}/{len(lit_results)} queries found papers")
    print(f"  Avg papers per query: {sum(r['papers'] for r in lit_results)/len(lit_results):.1f}")
    print(f"  Avg latency: {sum(r['latency_ms'] for r in lit_results)/len(lit_results):.0f}ms")
    
    drug_success = sum(1 for r in drug_results if r["found"])
    print(f"\n💊 Drug Agent:")
    print(f"  Success: {drug_success}/{len(drug_results)} drugs found")
    
    smiles_success = sum(1 for r in smiles_results if r["props"] > 0)
    print(f"\n🧪 SMILES Agent:")
    print(f"  Success: {smiles_success}/{len(smiles_results)} molecules analyzed")
    
    print("\n" + "=" * 60)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())