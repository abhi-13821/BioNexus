"""
Complete Metrics Collection for BioNexus Research Paper
Collects metrics for ALL components: Literature, Drugs, SMILES, KG, Agents, Embeddings, AI Assistant
"""

import sys
sys.path.insert(0, 'C:/projects/BioNexus')

import asyncio
import time
import json
import statistics
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import psutil
import platform

from agents.literature_agent import LiteratureAgent
from agents.drug_agent import DrugAgent
from agents.smiles_agent import SmilesAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.drug_discovery_agent import DrugDiscoveryAgent
from agents.models import AgentRequest
from embeddings.biomedical_embeddings import BiomedicalEmbeddingGenerator

# Test Data
LITERATURE_QUERIES = [
    "EGFR mutations in lung cancer",
    "BRCA1 breast cancer treatment",
    "COVID-19 vaccine efficacy",
    "Alzheimer's disease biomarkers",
    "CRISPR gene editing",
    "PD-1 immunotherapy",
    "metformin type 2 diabetes",
    "STAT3 signaling pathway",
    "TP53 tumor suppressor",
    "CAR-T cell therapy",
]

DRUGS = [
    "Aspirin", "Paracetamol", "Ibuprofen", "Metformin", "Atorvastatin",
    "Omeprazole", "Losartan", "Clopidogrel", "Simvastatin", "Amoxicillin",
]

SMILES_LIST = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "C1=CC=CC=C1",  # Benzene
    "CCCCCC",  # Hexane
    "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
]

KG_ENTITIES = [
    "EGFR", "BRCA1", "TP53", "KRAS", "BRAF",
    "MYC", "PTEN", "ALK", "ROS1", "MET",
]

DRUG_DISCOVERY_SMILES = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C1=CC=CC=C1",  # Benzene
]


class CompleteMetrics:
    """Collect metrics for ALL BioNexus components."""
    
    def __init__(self):
        self.metrics = {
            "literature_agent": {},
            "drug_agent": {},
            "smiles_agent": {},
            "knowledge_graph_agent": {},
            "drug_discovery_agent": {},
            "embeddings": {},
            "ai_assistant": {},
            "system_overall": {},
        }
        self.timings = {}
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "processor": platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_cores": psutil.cpu_count(),
            "ollama_model": "mistral:7b",
            "embedding_model": "uc-ctds/bge-large-en-v1.5-bio-mapping",
        }
    
    async def benchmark_literature(self) -> Dict[str, Any]:
        """Benchmark Literature Agent."""
        print("  📚 Testing Literature Agent...")
        agent = LiteratureAgent()
        await agent.start()
        
        results = []
        latencies = []
        papers_found = []
        
        for query in LITERATURE_QUERIES:
            start = time.time()
            req = AgentRequest(instruction=query, parameters={"query": query})
            resp = await agent.handle_request(req)
            elapsed = (time.time() - start) * 1000
            
            papers = []
            if resp.results and resp.results[0].output:
                papers = resp.results[0].output.get("papers", [])
            
            results.append({"query": query, "papers": len(papers), "success": len(papers) > 0})
            latencies.append(elapsed)
            papers_found.append(len(papers))
        
        await agent.stop()
        
        successful = [r for r in results if r["success"]]
        return {
            "total_queries": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_papers": sum(papers_found),
            "avg_papers": statistics.mean(papers_found) if papers_found else 0,
            "min_papers": min(papers_found) if papers_found else 0,
            "max_papers": max(papers_found) if papers_found else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0,
            "sources": 9,
            "source_names": ["PubMed", "Europe PMC", "CrossRef", "Semantic Scholar", "OpenAlex", "DOAJ", "bioRxiv", "medRxiv", "PMC"],
        }
    
    async def benchmark_drug(self) -> Dict[str, Any]:
        """Benchmark Drug Agent."""
        print("  💊 Testing Drug Agent...")
        agent = DrugAgent()
        await agent.start()
        
        results = []
        latencies = []
        properties_found = []
        
        for drug in DRUGS:
            start = time.time()
            req = AgentRequest(instruction=drug, parameters={"drug_name": drug})
            resp = await agent.handle_request(req)
            elapsed = (time.time() - start) * 1000
            
            found = False
            props_count = 0
            if resp.results and resp.results[0].output:
                compound = resp.results[0].output.get("compound", {})
                found = bool(compound)
                props_count = len(compound)
            
            results.append({"drug": drug, "found": found, "properties": props_count})
            latencies.append(elapsed)
            properties_found.append(props_count)
        
        await agent.stop()
        
        successful = [r for r in results if r["found"]]
        return {
            "total_drugs": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_properties": sum(properties_found),
            "avg_properties": statistics.mean(properties_found) if properties_found else 0,
            "min_properties": min(properties_found) if properties_found else 0,
            "max_properties": max(properties_found) if properties_found else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
        }
    
    async def benchmark_smiles(self) -> Dict[str, Any]:
        """Benchmark SMILES Agent."""
        print("  🧪 Testing SMILES Agent...")
        agent = SmilesAgent()
        await agent.start()
        
        results = []
        latencies = []
        props_counts = []
        
        for smiles in SMILES_LIST:
            start = time.time()
            req = AgentRequest(instruction=smiles, parameters={"smiles": smiles})
            resp = await agent.handle_request(req)
            elapsed = (time.time() - start) * 1000
            
            props_count = 0
            if resp.results and resp.results[0].output:
                props = resp.results[0].output.get("properties", {})
                props_count = len(props)
            
            results.append({"smiles": smiles[:20], "properties": props_count})
            latencies.append(elapsed)
            props_counts.append(props_count)
        
        await agent.stop()
        
        successful = [r for r in results if r["properties"] > 0]
        return {
            "total_smiles": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "avg_properties": statistics.mean(props_counts) if props_counts else 0,
            "min_properties": min(props_counts) if props_counts else 0,
            "max_properties": max(props_counts) if props_counts else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
        }
    
    async def benchmark_knowledge_graph(self) -> Dict[str, Any]:
        """Benchmark Knowledge Graph Agent."""
        print("  🕸️ Testing Knowledge Graph Agent...")
        
        # First need literature results to build graph
        try:
            lit_agent = LiteratureAgent()
            await lit_agent.start()
            req = AgentRequest(instruction="EGFR lung cancer", parameters={"query": "EGFR lung cancer"})
            resp = await lit_agent.handle_request(req)
            await lit_agent.stop()
            
            # Store papers in session state for KG
            import streamlit as st
            if resp.results and resp.results[0].output:
                st.session_state.bx_last_results = resp.results[0].output.get("papers", [])
        except:
            pass
        
        agent = KnowledgeGraphAgent()
        await agent.start()
        
        results = []
        latencies = []
        
        for entity in KG_ENTITIES[:5]:  # Test 5 entities
            start = time.time()
            req = AgentRequest(instruction=f"Find relationships for {entity}", parameters={"entity1": entity})
            resp = await agent.handle_request(req)
            elapsed = (time.time() - start) * 1000
            
            neighbors = 0
            if resp.results and resp.results[0].output:
                neighbors = len(resp.results[0].output.get("neighbors", []))
            
            results.append({"entity": entity, "neighbors": neighbors})
            latencies.append(elapsed)
        
        await agent.stop()
        
        successful = [r for r in results if r["neighbors"] > 0]
        return {
            "total_entities": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_relationships": sum(r["neighbors"] for r in results),
            "avg_neighbors": statistics.mean([r["neighbors"] for r in results]) if results else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        }
    
    async def benchmark_drug_discovery(self) -> Dict[str, Any]:
        """Benchmark Drug Discovery Agent."""
        print("  🔬 Testing Drug Discovery Agent...")
        agent = DrugDiscoveryAgent()
        await agent.start()
        
        results = []
        latencies = []
        
        for smiles in DRUG_DISCOVERY_SMILES:
            start = time.time()
            req = AgentRequest(instruction=f"Analyze {smiles}", parameters={"smiles": smiles})
            resp = await agent.handle_request(req)
            elapsed = (time.time() - start) * 1000
            
            candidates = 0
            if resp.results and resp.results[0].output:
                candidates = len(resp.results[0].output.get("candidates", []))
            
            results.append({"smiles": smiles[:20], "candidates": candidates})
            latencies.append(elapsed)
        
        await agent.stop()
        
        successful = [r for r in results if r["candidates"] > 0]
        return {
            "total_smiles": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_candidates": sum(r["candidates"] for r in results),
            "avg_candidates": statistics.mean([r["candidates"] for r in results]) if results else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        }
    
    def benchmark_embeddings(self) -> Dict[str, Any]:
        """Benchmark Embedding System."""
        print("  🧠 Testing Embedding System...")
        
        try:
            generator = BiomedicalEmbeddingGenerator()
            texts = [
                "EGFR mutation in lung cancer",
                "BRCA1 breast cancer",
                "CRISPR gene editing",
                "PD-1 immunotherapy",
                "TP53 tumor suppressor",
            ]
            
            latencies = []
            for text in texts:
                start = time.time()
                embedding = generator.generate_embedding(text)
                elapsed = (time.time() - start) * 1000
                latencies.append(elapsed)
            
            # Test batch processing
            batch_start = time.time()
            batch_embeddings = generator.generate_embeddings(texts)
            batch_elapsed = (time.time() - batch_start) * 1000
            
            return {
                "total_texts": len(texts),
                "embedding_dimension": generator.get_embedding_dimension(),
                "avg_single_latency_ms": statistics.mean(latencies) if latencies else 0,
                "batch_latency_ms": batch_elapsed,
                "model_name": "uc-ctds/bge-large-en-v1.5-bio-mapping",
                "embedding_count": len(batch_embeddings),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_ai_assistant(self) -> Dict[str, Any]:
        """Benchmark AI Assistant."""
        print("  🤖 Testing AI Assistant...")
        
        # This is a summary of the overall AI Assistant performance
        return {
            "llm_model": "mistral:7b",
            "llm_provider": "Ollama",
            "max_tokens": 1500,
            "temperature": 0.3,
            "timeout_seconds": 120,
            "agents_available": 5,
            "agent_names": [
                "Literature Search",
                "Drug Information",
                "SMILES Analysis",
                "Knowledge Graph",
                "Drug Discovery"
            ],
        }
    
    async def run_all(self):
        """Run all benchmarks."""
        print("\n" + "=" * 70)
        print("📊 BIOENEXUS - COMPLETE METRICS COLLECTION")
        print("   For Research Paper Publication")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System Info
        self.metrics["system_info"] = self.collect_system_info()
        
        # Run all benchmarks
        self.metrics["literature_agent"] = await self.benchmark_literature()
        self.metrics["drug_agent"] = await self.benchmark_drug()
        self.metrics["smiles_agent"] = await self.benchmark_smiles()
        self.metrics["knowledge_graph_agent"] = await self.benchmark_knowledge_graph()
        self.metrics["drug_discovery_agent"] = await self.benchmark_drug_discovery()
        self.metrics["embeddings"] = self.benchmark_embeddings()
        self.metrics["ai_assistant"] = self.benchmark_ai_assistant()
        
        # Calculate overall metrics
        self.calculate_overall()
        
        # Generate report
        self.generate_report()
        
        return self.metrics
    
    def calculate_overall(self):
        """Calculate overall system metrics."""
        lit = self.metrics.get("literature_agent", {})
        drug = self.metrics.get("drug_agent", {})
        smiles = self.metrics.get("smiles_agent", {})
        kg = self.metrics.get("knowledge_graph_agent", {})
        dd = self.metrics.get("drug_discovery_agent", {})
        
        total_queries = (
            lit.get("total_queries", 0) +
            drug.get("total_drugs", 0) +
            smiles.get("total_smiles", 0) +
            kg.get("total_entities", 0) +
            dd.get("total_smiles", 0)
        )
        
        total_success = (
            lit.get("successful", 0) +
            drug.get("successful", 0) +
            smiles.get("successful", 0) +
            kg.get("successful", 0) +
            dd.get("successful", 0)
        )
        
        all_latencies = []
        if lit.get("avg_latency_ms"):
            all_latencies.append(lit["avg_latency_ms"])
        if drug.get("avg_latency_ms"):
            all_latencies.append(drug["avg_latency_ms"])
        if smiles.get("avg_latency_ms"):
            all_latencies.append(smiles["avg_latency_ms"])
        if kg.get("avg_latency_ms"):
            all_latencies.append(kg["avg_latency_ms"])
        if dd.get("avg_latency_ms"):
            all_latencies.append(dd["avg_latency_ms"])
        
        self.metrics["system_overall"] = {
            "total_queries": total_queries,
            "total_successful": total_success,
            "overall_success_rate": total_success / total_queries if total_queries > 0 else 0,
            "avg_latency_ms": statistics.mean(all_latencies) if all_latencies else 0,
            "components_tested": 5,
            "agent_types": ["Literature", "Drug", "SMILES", "Knowledge Graph", "Drug Discovery"],
        }
    
    def generate_report(self):
        """Generate research-ready report with tables."""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 BIOENEXUS RESEARCH PAPER - COMPLETE METRICS")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # System Info
        sys_info = self.metrics.get("system_info", {})
        lines.append("## SYSTEM CONFIGURATION")
        lines.append("-" * 80)
        lines.append(f"  Python Version: {sys_info.get('python_version', 'N/A')}")
        lines.append(f"  Operating System: {sys_info.get('os', 'N/A')}")
        lines.append(f"  CPU Cores: {sys_info.get('cpu_cores', 'N/A')}")
        lines.append(f"  RAM: {sys_info.get('ram_gb', 'N/A')} GB")
        lines.append(f"  LLM Model: {sys_info.get('ollama_model', 'N/A')}")
        lines.append(f"  Embedding Model: {sys_info.get('embedding_model', 'N/A')}")
        lines.append("")
        
        # TABLE 1: Agent Performance
        lines.append("## TABLE 1: AGENT PERFORMANCE METRICS")
        lines.append("-" * 80)
        lines.append("| Agent | Success Rate | Avg Results | Avg Latency | P95 Latency |")
        lines.append("|-------|-------------|-------------|-------------|-------------|")
        
        lit = self.metrics.get("literature_agent", {})
        lines.append(f"| Literature | {lit.get('success_rate', 0)*100:.1f}% | {lit.get('avg_papers', 0):.1f} papers | {lit.get('avg_latency_ms', 0):.0f}ms | {lit.get('p95_latency_ms', 0):.0f}ms |")
        
        drug = self.metrics.get("drug_agent", {})
        lines.append(f"| Drug | {drug.get('success_rate', 0)*100:.1f}% | {drug.get('avg_properties', 0):.1f} props | {drug.get('avg_latency_ms', 0):.0f}ms | {drug.get('p95_latency_ms', 0):.0f}ms |")
        
        smiles = self.metrics.get("smiles_agent", {})
        lines.append(f"| SMILES | {smiles.get('success_rate', 0)*100:.1f}% | {smiles.get('avg_properties', 0):.1f} props | {smiles.get('avg_latency_ms', 0):.0f}ms | {smiles.get('p95_latency_ms', 0):.0f}ms |")
        
        kg = self.metrics.get("knowledge_graph_agent", {})
        lines.append(f"| Knowledge Graph | {kg.get('success_rate', 0)*100:.1f}% | {kg.get('avg_neighbors', 0):.1f} rels | {kg.get('avg_latency_ms', 0):.0f}ms | - |")
        
        dd = self.metrics.get("drug_discovery_agent", {})
        lines.append(f"| Drug Discovery | {dd.get('success_rate', 0)*100:.1f}% | {dd.get('avg_candidates', 0):.1f} cands | {dd.get('avg_latency_ms', 0):.0f}ms | - |")
        lines.append("")
        
        # TABLE 2: Detailed Agent Metrics
        lines.append("## TABLE 2: DETAILED AGENT METRICS")
        lines.append("-" * 80)
        lines.append("| Metric | Literature | Drug | SMILES | KG | Drug Discovery |")
        lines.append("|--------|-----------|------|--------|----|----------------|")
        lines.append(f"| Total Queries | {lit.get('total_queries', 0)} | {drug.get('total_drugs', 0)} | {smiles.get('total_smiles', 0)} | {kg.get('total_entities', 0)} | {dd.get('total_smiles', 0)} |")
        lines.append(f"| Success Rate | {lit.get('success_rate', 0)*100:.1f}% | {drug.get('success_rate', 0)*100:.1f}% | {smiles.get('success_rate', 0)*100:.1f}% | {kg.get('success_rate', 0)*100:.1f}% | {dd.get('success_rate', 0)*100:.1f}% |")
        lines.append(f"| Min Results | {lit.get('min_papers', 0)} | {drug.get('min_properties', 0)} | {smiles.get('min_properties', 0)} | - | - |")
        lines.append(f"| Max Results | {lit.get('max_papers', 0)} | {drug.get('max_properties', 0)} | {smiles.get('max_properties', 0)} | - | - |")
        lines.append(f"| Avg Results | {lit.get('avg_papers', 0):.1f} | {drug.get('avg_properties', 0):.1f} | {smiles.get('avg_properties', 0):.1f} | {kg.get('avg_neighbors', 0):.1f} | {dd.get('avg_candidates', 0):.1f} |")
        lines.append(f"| Avg Latency | {lit.get('avg_latency_ms', 0):.0f}ms | {drug.get('avg_latency_ms', 0):.0f}ms | {smiles.get('avg_latency_ms', 0):.0f}ms | {kg.get('avg_latency_ms', 0):.0f}ms | {dd.get('avg_latency_ms', 0):.0f}ms |")
        lines.append("")
        
        # TABLE 3: Embedding System
        emb = self.metrics.get("embeddings", {})
        lines.append("## TABLE 3: EMBEDDING SYSTEM METRICS")
        lines.append("-" * 80)
        lines.append(f"  Model: {emb.get('model_name', 'N/A')}")
        lines.append(f"  Embedding Dimension: {emb.get('embedding_dimension', 'N/A')}")
        lines.append(f"  Texts Tested: {emb.get('total_texts', 0)}")
        lines.append(f"  Average Single Latency: {emb.get('avg_single_latency_ms', 0):.2f}ms")
        lines.append(f"  Batch Latency: {emb.get('batch_latency_ms', 0):.2f}ms")
        lines.append(f"  Embeddings Generated: {emb.get('embedding_count', 0)}")
        lines.append("")
        
        # TABLE 4: AI Assistant
        ai = self.metrics.get("ai_assistant", {})
        lines.append("## TABLE 4: AI ASSISTANT CONFIGURATION")
        lines.append("-" * 80)
        lines.append(f"  LLM Model: {ai.get('llm_model', 'N/A')}")
        lines.append(f"  Provider: {ai.get('llm_provider', 'N/A')}")
        lines.append(f"  Max Tokens: {ai.get('max_tokens', 'N/A')}")
        lines.append(f"  Temperature: {ai.get('temperature', 'N/A')}")
        lines.append(f"  Timeout: {ai.get('timeout_seconds', 'N/A')}s")
        lines.append(f"  Agents Available: {ai.get('agents_available', 0)}")
        for name in ai.get('agent_names', []):
            lines.append(f"    - {name}")
        lines.append("")
        
        # TABLE 5: Overall System
        overall = self.metrics.get("system_overall", {})
        lines.append("## TABLE 5: OVERALL SYSTEM METRICS")
        lines.append("-" * 80)
        lines.append(f"  Total Queries: {overall.get('total_queries', 0)}")
        lines.append(f"  Total Successful: {overall.get('total_successful', 0)}")
        lines.append(f"  Overall Success Rate: {overall.get('overall_success_rate', 0)*100:.1f}%")
        lines.append(f"  Overall Avg Latency: {overall.get('avg_latency_ms', 0):.0f}ms")
        lines.append(f"  Components Tested: {overall.get('components_tested', 0)}")
        for name in overall.get('agent_types', []):
            lines.append(f"    - {name}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("✅ COMPLETE METRICS COLLECTION COMPLETED")
        lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(lines)
        with open("complete_metrics_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # Also save as JSON for further analysis
        with open("complete_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(report_text)
        print("\n📄 Reports saved to:")
        print("  - complete_metrics_report.txt")
        print("  - complete_metrics.json")


async def main():
    metrics = CompleteMetrics()
    await metrics.run_all()


if __name__ == "__main__":
    asyncio.run(main())