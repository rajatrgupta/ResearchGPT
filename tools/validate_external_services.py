import sys
import os
import time
import inspect
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Inspect Tavily API for timeout support
print("--- Inspecting TavilyClient ---")
try:
    from tavily import TavilyClient
    client_init_sig = inspect.signature(TavilyClient.__init__)
    search_sig = inspect.signature(TavilyClient.search)
    print("TavilyClient.__init__ signature:", client_init_sig)
    print("TavilyClient.search signature:", search_sig)
except ImportError:
    print("tavily-python not installed.")
except Exception as e:
    print(f"Error inspecting TavilyClient: {e}")

print("\n--- Test A: Tavily Search ---")
try:
    from app.search import get_search_provider
    provider = get_search_provider()
    print(f"Active Provider: {provider.name}")
    
    start_time = time.time()
    results = provider.search("Test query for Tavily timeout validation", max_results=1)
    end_time = time.time()
    
    print(f"Start Timestamp: {start_time}")
    print(f"End Timestamp: {end_time}")
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
    print(f"Success/Failure: Success (Found {len(results)} results)")
except Exception as e:
    end_time = time.time()
    print(f"End Timestamp: {end_time}")
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
    print(f"Success/Failure: Failure")
    print(f"Exception: {type(e).__name__}: {str(e)}")


print("\n--- Test B: OpenRouter LLM ---")
try:
    from app.core.llm import get_llm
    llm_info = get_llm()
    llm = llm_info["llm"]
    print(f"LLM Config: {llm_info['model_used']}")
    
    start_time = time.time()
    print(f"Start Timestamp: {start_time}")
    # We use invoke with a simple human message
    response = llm.invoke("Reply with OK")
    end_time = time.time()
    
    print(f"End Timestamp: {end_time}")
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
    print(f"Success/Failure: Success")
    print(f"Response: {response.content}")
    # Determine which model actually responded
    model_name = response.response_metadata.get('model_name', 'Unknown model')
    print(f"Model that responded: {model_name}")
except Exception as e:
    end_time = time.time()
    print(f"End Timestamp: {end_time}")
    print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
    print(f"Success/Failure: Failure")
    print(f"Exception: {type(e).__name__}: {str(e)}")
