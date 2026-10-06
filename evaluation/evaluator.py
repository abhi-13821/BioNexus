"""
Evaluator for BioNexus Agents
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from metrics import Metrics
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate agent performance."""
    
    def __init__(self):
        self.metrics = Metrics()
        self.results = {}
    
    async def evaluate_literature_agent(self, agent, test_queries: List[str]) -> Dict[str, Any]:
        """Evaluate Literature Agent performance."""
        logger.info(f"Evaluating Literature Agent with {len(test_queries)} queries...")
        
        results = {
            "agent": "literature_search_agent",
            "queries": [],
            "latency": [],
            "success_rate": 0,
            "average_papers": 0
        }
        
        for query in test_queries:
            try:
                start_time = datetime.now()
                response = await agent.handle_request(
                    AgentRequest(instruction=query, parameters={"query": query})
                )
                end_time = datetime.now()
                
                latency = (end_time - start_time).total_seconds() * 1000
                
                papers = []
                if response.results and response.results[0].output:
                    papers = response.results[0].output.get("papers", [])
                
                results["queries"].append({
                    "query": query,
                    "papers_found": len(papers),
                    "success": len(papers) > 0,
                    "latency_ms": latency
                })
                results["latency"].append(latency)
                
            except Exception as e:
                logger.error(f"Query failed: {query}, Error: {e}")
                results["queries"].append({
                    "query": query,
                    "papers_found": 0,
                    "success": False,
                    "latency": 0,
                    "error": str(e)
                })
        
        # Calculate statistics
        successful = [q for q in results["queries"] if q.get("success", False)]
        results["success_rate"] = len(successful) / len(test_queries) if test_queries else 0
        results["average_papers"] = sum(q.get("papers_found", 0) for q in results["queries"]) / len(test_queries) if test_queries else 0
        results["average_latency_ms"] = sum(results["latency"]) / len(results["latency"]) if results["latency"] else 0
        
        return results
    
    async def evaluate_drug_agent(self, agent, test_drugs: List[str]) -> Dict[str, Any]:
        """Evaluate Drug Agent performance."""
        logger.info(f"Evaluating Drug Agent with {len(test_drugs)} drugs...")
        
        results = {
            "agent": "drug_info_agent",
            "drugs": [],
            "success_rate": 0,
            "average_properties": 0
        }
        
        for drug in test_drugs:
            try:
                response = await agent.handle_request(
                    AgentRequest(instruction=drug, parameters={"drug_name": drug})
                )
                
                drug_data = {}
                if response.results and response.results[0].output:
                    drug_data = response.results[0].output
                
                compound = drug_data.get("compound", {})
                results["drugs"].append({
                    "drug": drug,
                    "found": bool(compound),
                    "properties": list(compound.keys()) if compound else []
                })
                
            except Exception as e:
                logger.error(f"Drug query failed: {drug}, Error: {e}")
                results["drugs"].append({
                    "drug": drug,
                    "found": False,
                    "error": str(e)
                })
        
        found = [d for d in results["drugs"] if d.get("found", False)]
        results["success_rate"] = len(found) / len(test_drugs) if test_drugs else 0
        results["average_properties"] = sum(len(d.get("properties", [])) for d in results["drugs"]) / len(test_drugs) if test_drugs else 0
        
        return results
    
    async def evaluate_smiles_agent(self, agent, test_smiles: List[str]) -> Dict[str, Any]:
        """Evaluate SMILES Agent performance."""
        logger.info(f"Evaluating SMILES Agent with {len(test_smiles)} SMILES strings...")
        
        results = {
            "agent": "smiles_analysis_agent",
            "smiles": [],
            "success_rate": 0
        }
        
        for smiles in test_smiles:
            try:
                response = await agent.handle_request(
                    AgentRequest(instruction=smiles, parameters={"smiles": smiles})
                )
                
                output = response.results[0].output if response.results else {}
                properties = output.get("properties", {})
                
                results["smiles"].append({
                    "smiles": smiles,
                    "success": bool(properties),
                    "properties_count": len(properties)
                })
                
            except Exception as e:
                logger.error(f"SMILES analysis failed: {smiles}, Error: {e}")
                results["smiles"].append({
                    "smiles": smiles,
                    "success": False,
                    "error": str(e)
                })
        
        successful = [s for s in results["smiles"] if s.get("success", False)]
        results["success_rate"] = len(successful) / len(test_smiles) if test_smiles else 0
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "📊 BIOENEXUS EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        for agent_name, agent_results in results.items():
            if not agent_results:
                continue
            
            lines.append(f"\n📚 AGENT: {agent_name}")
            lines.append("-" * 40)
            
            if agent_name == "literature_search_agent":
                lines.append(f"  Queries tested: {len(agent_results.get('queries', []))}")
                lines.append(f"  Success rate: {agent_results.get('success_rate', 0) * 100:.1f}%")
                lines.append(f"  Average papers per query: {agent_results.get('average_papers', 0):.1f}")
                lines.append(f"  Average latency: {agent_results.get('average_latency_ms', 0):.1f} ms")
            
            elif agent_name == "drug_info_agent":
                lines.append(f"  Drugs tested: {len(agent_results.get('drugs', []))}")
                lines.append(f"  Success rate: {agent_results.get('success_rate', 0) * 100:.1f}%")
                lines.append(f"  Average properties per drug: {agent_results.get('average_properties', 0):.1f}")
            
            elif agent_name == "smiles_analysis_agent":
                lines.append(f"  SMILES tested: {len(agent_results.get('smiles', []))}")
                lines.append(f"  Success rate: {agent_results.get('success_rate', 0) * 100:.1f}%")
        
        lines.append("\n" + "=" * 60)
        lines.append("✅ EVALUATION COMPLETE")
        lines.append("=" * 60)
        
        return "\n".join(lines)