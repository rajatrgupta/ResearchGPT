"""
LangGraph Workflow Orchestration for DeepTrace

This module compiles the individual agent nodes (Planner, Search, Retriever, Critic, Writer) 
into a cohesive, executable Directed Acyclic Graph (DAG) with conditional retry loops.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from app.graph.state import ResearchState
from app.agents.planner import planner_node
from app.agents.search import search_node
from app.agents.retriever import retriever_node
from app.agents.critic import critic_node
from app.agents.query_optimizer import query_optimizer_node
from app.agents.writer import writer_node

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
# Hard limit on research retries to prevent infinite loops and 
# excessive API costs.
MAX_ITERATIONS = 3

# ==========================================
# RULE 1: WHAT IS THIS CONCEPT?
# ==========================================
# This is the orchestration layer. We are taking isolated python functions 
# (our nodes) and wiring them together using Edges and Conditional Routing.
#
# PHASE 3 ASCII WORKFLOW DIAGRAM:
#
#       [START]
#          |
#          v
#     +---------+
#     | Planner |   
#     +---------+
#          |
#          v
#     +---------+ <─────────────────┐
#     | Search  |                   │ (Retry Path)
#     +---------+                   │
#          |                        │
#          v                        │
#     +-----------+                 │
#     | Retriever |                 │
#     +-----------+                 │
#          |                        │
#          v                        │
#     +-----------+                 │
#     |  Critic   | ────────────────┘
#     +-----------+
#          |
#          ├─ [is_valid == True] ───┐
#          |                        |
#          └─ [iterations >= MAX] ──┤
#                                   |
#                                   | [REJECTED and < MAX]
#                                   v
#                           +-----------------+
#                           | QueryOptimizer  |
#                           +-----------------+
#                                   |
#                                   v (Back to Search)

def route_after_critic(state: ResearchState) -> Literal["query_optimizer", "writer", "force_degradation"]:
    """
    Conditional routing function that decides the next step in the graph.
    
    Logic:
    - If Critic approved (is_valid == True): Proceed to Writer.
    - If Max Iterations reached: Force proceed to Writer via degradation node.
    - Otherwise: Route to Query Optimizer to craft new search paths.
    """
    
    is_valid = state.get("is_valid", False)
    iteration_count = state.get("iteration_count", 0)
    
    if is_valid:
        print(f"\n[Workflow] Critic approved (Score: {state.get('quality_score')}). Moving to Writer.")
        return "writer"
        
    if iteration_count >= MAX_ITERATIONS:
        print(f"\n[Workflow] Critic rejected but MAX_ITERATIONS ({MAX_ITERATIONS}) reached. Forcing synthesis.")
        return "force_degradation"
        
    print(f"\n[Workflow] Critic rejected (Score: {state.get('quality_score')}). Feedback: {state.get('critic_feedback')}")
    print(f"[Workflow] Retry loop {iteration_count}/{MAX_ITERATIONS} triggered. Routing to Query Optimizer.")
    return "query_optimizer"


def force_degradation_node(state: ResearchState) -> dict:
    """Sets the degradation mode flag to true before routing to writer."""
    return {"degradation_mode": True}

def build_graph():
    """
    Builds and compiles the LangGraph StateGraph for the DeepTrace research workflow.
    
    Returns:
        CompiledGraph: An executable LangGraph object.
    """
    
    # 1. Initialize the StateGraph with our strictly typed ResearchState
    workflow = StateGraph(ResearchState)
    
    # 2. Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("query_optimizer", query_optimizer_node)
    workflow.add_node("force_degradation", force_degradation_node)
    workflow.add_node("writer", writer_node)
    
    # 3. Define the Flow (Edges)
    # Standard linear entry path
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "search")
    workflow.add_edge("search", "retriever")
    workflow.add_edge("retriever", "critic")
    
    # RAG UPGRADE: Conditional Edge
    # Instead of a direct line to 'writer', we insert a dynamic router.
    # The 'route_after_critic' function inspects the state and returns 
    # either 'search' or 'writer'.
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "query_optimizer": "query_optimizer",
            "writer": "writer",
            "force_degradation": "force_degradation"
        }
    )
    workflow.add_edge("query_optimizer", "search")
    workflow.add_edge("force_degradation", "writer")
    
    # Standard exit path
    workflow.add_edge("writer", END)
    
    # 4. Compile the Graph
    compiled_graph = workflow.compile()
    
    return compiled_graph
