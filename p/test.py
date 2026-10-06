# test.py
import asyncio
import logging
from agents.smart_coordinator import SmartCoordinator

# Turn off warnings for cleaner output
logging.basicConfig(level=logging.ERROR)

async def test():
    print("Creating coordinator...")
    coordinator = SmartCoordinator()
    
    print("Initializing...")
    await coordinator.initialize()
    
    print("\n" + "="*50)
    print("TEST 1: Literature Search")
    print("="*50)
    result = await coordinator.process_query("Find papers about cancer immunotherapy")
    print(result.get("message", "No message"))
    print(f"\nPapers found: {len(result.get('data', {}).get('papers', []))}")

asyncio.run(test())