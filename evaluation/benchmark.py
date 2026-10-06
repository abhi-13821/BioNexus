"""
Benchmark Script for BioNexus
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # ✅ Add evaluation folder to path

from agents.literature_agent import LiteratureAgent
from agents.drug_agent import DrugAgent
from agents.smiles_agent import SmilesAgent
from agents.models import AgentRequest
from evaluator import Evaluator  # ✅ Now works because evaluation folder is in path
from test_data import LITERATURE_TEST_QUERIES, DRUG_TEST_LIST, SMILES_TEST_LIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_benchmark():
    """Run the complete benchmark."""
    print("\n" + "=" * 60)
    print("🚀 RUNNING BIOENEXUS BENCHMARK")
    print("=" * 60)
    
    evaluator = Evaluator()
    results = {}
    
    # 1. Test Literature Agent
    print("\n📚 Testing Literature Agent...")
    try:
        agent = LiteratureAgent()
        await agent.start()
        results["literature_search_agent"] = await evaluator.evaluate_literature_agent(
            agent, LITERATURE_TEST_QUERIES
        )
        await agent.stop()
    except Exception as e:
        logger.error(f"Literature Agent benchmark failed: {e}")
        results["literature_search_agent"] = {"error": str(e)}
    
    # 2. Test Drug Agent
    print("\n💊 Testing Drug Agent...")
    try:
        agent = DrugAgent()
        await agent.start()
        results["drug_info_agent"] = await evaluator.evaluate_drug_agent(
            agent, DRUG_TEST_LIST
        )
        await agent.stop()
    except Exception as e:
        logger.error(f"Drug Agent benchmark failed: {e}")
        results["drug_info_agent"] = {"error": str(e)}
    
    # 3. Test SMILES Agent
    print("\n🧪 Testing SMILES Agent...")
    try:
        agent = SmilesAgent()
        await agent.start()
        results["smiles_analysis_agent"] = await evaluator.evaluate_smiles_agent(
            agent, SMILES_TEST_LIST
        )
        await agent.stop()
    except Exception as e:
        logger.error(f"SMILES Agent benchmark failed: {e}")
        results["smiles_analysis_agent"] = {"error": str(e)}
    
    # Generate report
    print("\n" + evaluator.generate_report(results))
    
    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(evaluator.generate_report(results))
    print("\n📄 Report saved to: evaluation_report.txt")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark())