echo 'import sys
import asyncio
sys.path.insert(0, "C:/projects/BioNexus")

from agents.literature_agent import LiteratureAgent
from agents.models import AgentRequest

async def test():
    agent = LiteratureAgent()
    await agent.start()
    
    request = AgentRequest(
        instruction="EGFR lung cancer",
        parameters={"query": "EGFR lung cancer"}
    )
    response = await agent.handle_request(request)
    
    print("Status:", response.status)
    print("Results count:", len(response.results))
    
    if response.results:
        output = response.results[0].output
        print("Output keys:", list(output.keys()))
        papers = output.get("papers", [])
        print("Papers found:", len(papers))
        if papers:
            print("First paper:", papers[0].get("title", "No title")[:100])
    else:
        print("No results")

asyncio.run(test())' > test_agent.py