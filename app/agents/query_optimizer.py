"""
Query Optimizer Agent for DeepTrace

This module defines the Query Optimizer Node. It acts when the Critic rejects
the retrieved documents. It analyzes the original query, the Critic's feedback,
and the previously retrieved documents to formulate highly targeted search queries.
"""

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from app.graph.state import ResearchState
from app.core.llm import get_llm

# ==========================================
# PYDANTIC SCHEMA
# ==========================================
class QueryOptimizationResult(BaseModel):
    retry_queries: List[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 distinct, search-engine friendly queries targeting the gaps identified by the Critic. Do not duplicate searches for information already found."
    )

def format_retrieved_documents_for_optimizer(documents: list[Dict[str, Any]]) -> str:
    """
    Helper function to format documents for the Optimizer's prompt.
    Extracts only title, snippet, and source to save token context and latency.
    """
    formatted_context = ""
    for i, doc in enumerate(documents):
        title = doc.get("title", "Untitled")
        snippet = doc.get("snippet", "")
        source = doc.get("source", "No Source")
        
        formatted_context += f"--- Document [{i+1}] ---\n"
        formatted_context += f"Title: {title}\n"
        formatted_context += f"Source: {source}\n"
        formatted_context += f"Snippet: {snippet}\n\n"
        
    return formatted_context

def query_optimizer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the query optimization phase during a retry loop.
    
    Args:
        state: The current ResearchState containing 'query', 'critic_feedback', 
               and 'retrieved_documents'.
               
    Returns:
        A dictionary containing the generated 'retry_queries', 'status', 
        and optionally 'errors'.
    """
    query = state.get("query", "")
    critic_feedback = state.get("critic_feedback", "")
    retrieved_documents = state.get("retrieved_documents", [])
    
    # Format the minimal context for the LLM
    context_str = format_retrieved_documents_for_optimizer(retrieved_documents)
    
    prompt = PromptTemplate(
        template="""You are an expert Search Query Optimizer.
The system attempted to research a topic but the QA Critic rejected the findings.
Your job is to generate EXACTLY 3 highly optimized DuckDuckGo search queries to find the missing information.

Original User Query: {query}

Critic Feedback (What is missing/wrong): {critic_feedback}

Previously Retrieved Documents (Do not search for what we already know):
{context}

Rules for new queries:
1. Output EXACTLY 3 queries.
2. No duplicates.
3. Use search-engine friendly phrasing (keywords, no conversational filler words).
4. Each query should target a different gap identified by the Critic.
""",
        input_variables=["query", "critic_feedback", "context"]
    )
    
    # Initialize the centralized, fault-tolerant LLM
    llm_config = get_llm()
    robust_llm = llm_config["llm"]

    # Bind our Pydantic schema to the LLM to enforce exactly 3 string outputs
    structured_llm = robust_llm.with_structured_output(QueryOptimizationResult)
    
    chain = prompt | structured_llm
    
    try:
        result: QueryOptimizationResult = chain.invoke({
            "query": query,
            "critic_feedback": critic_feedback,
            "context": context_str
        })
        
        return {
            "retry_queries": result.retry_queries,
            "status": "optimization_complete"
        }
        
    except Exception as e:
        error_msg = f"Query Optimizer failed during execution: {str(e)}"
        print(f"Warning: {error_msg}")
        # Deterministic fallback ensuring exactly 3 distinct search queries
        fallback_queries = [
            query,
            f"{query} analysis",
            f"{query} latest research"
        ]
        return {
            "retry_queries": fallback_queries,
            "errors": [error_msg],
            "status": "optimization_error"
        }
