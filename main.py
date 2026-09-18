"""
DeepTrace Main Execution Script

This script serves as the entry point to run our LangGraph multi-agent research pipeline.
"""

# IMPORTANT: Load environment variables FIRST.
from dotenv import load_dotenv
load_dotenv()

import uuid
import urllib.parse

from app.graph.workflow import build_graph, MAX_ITERATIONS
from app.vectorstore.qdrant_store import reset_collection

# ==========================================
# EXPLANATION OF INTERNAL NODE OPERATIONS
# ==========================================
# 1. PLANNER: Generates 5 sub-questions.
# 2. SEARCH: Scrapes web (uses Critic feedback on retries).
# 3. RETRIEVER: Indexes and pulls Top-K relevant facts.
# 4. CRITIC: Scores the research (0.0-1.0).
# 5. ROUTER: Determines if we should RETRY search or move to WRITER.
# 6. OPTIMIZER: (Only on Retry) Generates 3 highly targeted search strings.
# 7. WRITER: Synthesizes final markdown report.

def merge_state(current: dict, update: dict) -> dict:
    """
    Manually merges partial state updates, mimicking LangGraph's
    reducer logic for specific fields defined in ResearchState.
    """
    new_state = current.copy()
    for key, value in update.items():
        # Fields using operator.add require list extension
        if key in ["search_results", "sources"] and key in new_state:
            # Create a new list to avoid mutating the original reference
            new_state[key] = new_state[key] + value
        else:
            # Fields like quality_score, report, and iteration_count are replacements
            new_state[key] = value
    return new_state

def main():
    print("Resetting Qdrant collection for a fresh run...")
    reset_collection("deeptrace_research")

    # Phase 4B: Generate a unique run_id for this execution.
    # This UUID tags every document stored in Qdrant during this run, enabling
    # retrieval to be filtered to only this run's documents (run-level isolation).
    # It never changes during the lifetime of this main() call.
    run_id = str(uuid.uuid4())
    print(f"[DeepTrace] Run ID: {run_id[:8]}...  (used for Qdrant isolation)")

    print("Building DeepTrace Graph...")
    graph = build_graph()

    # 1. Create the initial state
    initial_state = {
        "run_id": run_id,
        "query": "Future of Artificial Intelligence",
        "sub_questions": [],
        "search_results": [],
        "retry_queries": [],
        "retrieved_documents": [],
        "sources": [],
        "report": "",
        "quality_score": 0.0,
        "iteration_count": 0,
        "status": "initialized",
        "errors": [],
        "critic_feedback": "",
        "is_valid": False
    }

    print(f"\n[DeepTrace] Starting research on: '{initial_state['query']}'")
    print("================================================================")
    print("Execution in progress. This may take 30-120 seconds...\n")

    try:
        # 2. Run the graph using STREAMING for real-time observability
        current_state = initial_state

        for output in graph.stream(initial_state):
            # output is a dict: {node_name: state_update}
            for node_name, state_update in output.items():

                # Logic for Iteration/Cycle headers
                if node_name == "planner":
                    print(f"--- CYCLE #1 ---")
                elif node_name == "search" and current_state["iteration_count"] >= 1:
                    print(f"\n--- CYCLE #{current_state['iteration_count'] + 1} ---")

                print(f"[{node_name.capitalize()}] Node Started...")

                # PRODUCTION FIX: Correctly merge partial state updates
                current_state = merge_state(current_state, state_update)

                # Logic for Critic/Router Visibility
                if node_name == "retriever":
                    docs = state_update.get("retrieved_documents", [])
                    print(f"[Retriever] Curated Top {len(docs)} documents after Trust-Tier Reranking:")
                    for idx, doc in enumerate(docs, 1):
                        url = doc.get("source") or doc.get("link") or "unknown"
                        domain = urllib.parse.urlparse(url).netloc if url != "unknown" else "unknown"
                        if domain.startswith("www."):
                            domain = domain[4:]
                        tier = doc.get("trust_tier", "DEFAULT_TRUST")
                        o_score = doc.get("original_score", 0.0)
                        r_score = doc.get("reranked_score", 0.0)
                        print(f"  {idx}. [{tier}] {domain} (Original: {o_score:.4f} -> Reranked: {r_score:.4f})")

                elif node_name == "critic":
                    score = state_update.get("quality_score", 0.0)
                    feedback = state_update.get("critic_feedback", "N/A")
                    is_valid = state_update.get("is_valid", False)

                    print(f"[Critic] Score: {score} | Feedback: {feedback}")

                    if is_valid:
                        print(f"[Router] Decision: APPROVED -> Moving to Writer.")
                    elif current_state["iteration_count"] >= MAX_ITERATIONS:
                        print(f"[Router] Decision: REJECTED | MAX_ITERATIONS ({MAX_ITERATIONS}) reached. Forcing Synthesis.")
                    else:
                        print(f"[Router] Decision: RETRY -> Routing to Query Optimizer.")

                elif node_name == "query_optimizer":
                    optimized_queries = state_update.get("retry_queries", [])
                    print(f"[QueryOptimizer] Generated {len(optimized_queries)} targeted queries.")

                print(f"[{node_name.capitalize()}] Node Complete.")

        # 3. Final Execution Summary
        print("\n--- DEEPTRACE EXECUTION COMPLETE ---")
        print(f"Final Status   : {current_state['status']}")
        print(f"Run ID         : {current_state.get('run_id', 'N/A')[:8]}...")
        print(f"Search Loops   : {current_state['iteration_count']}")
        print(f"Sources Found  : {len(current_state['sources'])}")
        print(f"Docs Retrieved : {len(current_state.get('retrieved_documents', []))}")
        print(f"Final Quality  : {current_state.get('quality_score', 0.0)}")
        print(f"Final Decision : {'APPROVED' if current_state.get('is_valid') else 'REJECTED/FORCED'}")
        print("------------------------------------\n")

        print("FINAL REPORT:")
        print("================================================================\n")
        print(current_state["report"])
        print("\n================================================================")

        if current_state["errors"]:
            print("\nWARNING - Non-fatal errors occurred during execution:")
            for error in current_state["errors"]:
                print(f"- {error}")

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline crashed: {str(e)}")

if __name__ == "__main__":
    main()
