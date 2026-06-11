"""
Retriever Agent for DeepTrace

This module defines the Retriever Node. Its responsibility is to bridge the gap
between the raw data gathered by the Search Agent and the synthesis performed
by the Writer Agent. It commits the raw search results to the Qdrant Vector DB
for long-term memory, and then immediately performs a semantic similarity search
to extract only the top most relevant chunks to answer the user's overarching query.
"""

from typing import Dict, Any
from app.graph.state import ResearchState
from app.vectorstore.qdrant_store import store_documents, search_similar

# ==========================================
# RULE 1: WHAT IS THIS CONCEPT?
# ==========================================
# The Retriever Agent is the "R" in RAG (Retrieval-Augmented Generation).
# 
# Why it exists:
# The Search Agent might scrape 50 different snippets across 5 sub-questions.
# We cannot feed all 50 snippets to the Writer LLM because:
# 1. It exceeds token limits (costly and errors out).
# 2. LLMs suffer from the "Lost in the Middle" phenomenon where they ignore 
#    data buried deep in massive prompts.
# 3. Much of the scraped data is noise or loosely related.
#
# Solution:
# We store all 50 snippets in Qdrant (building an index). We then take the 
# user's original overarching query (e.g., "Future of AI") and ask Qdrant: 
# "Which 5 snippets are mathematically closest in meaning to this query?"
# Qdrant returns only the pure, highly relevant facts.

def retriever_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the vector storage and retrieval phase of the research workflow.
    
    Args:
        state: The current ResearchState containing 'query' and 'search_results'.
        
    Returns:
        A dictionary containing the semantically filtered 'retrieved_documents',
        a 'status' indicator, and optionally 'errors'.
    """
    
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    
    # Defensive check: If we have no data, we cannot store or retrieve anything.
    if not search_results:
        return {
            "retrieved_documents": [],
            "errors": ["Retriever Node received empty search_results list."],
            "status": "retrieval_failed"
        }
        
    if not query:
        return {
             "retrieved_documents": [],
             "errors": ["Retriever Node requires a user query to perform similarity search."],
             "status": "retrieval_failed"
        }

    accumulated_errors = []
    retrieved_docs = []

    try:
        # STEP 1: Storage (Long-Term Memory)
        # We commit all the raw findings into Qdrant. This ensures the data 
        # is embedded and indexed for future semantic search.
        store_documents(documents=search_results, collection_name="deeptrace_research")
        
    except Exception as e:
        error_msg = f"Retriever Node failed during storage phase: {str(e)}"
        accumulated_errors.append(error_msg)
        print(error_msg)
        # If storage fails, we cannot retrieve. We must fail gracefully.
        return {
            "retrieved_documents": [],
            "errors": accumulated_errors,
            "status": "retrieval_failed"
        }

    try:
        # STEP 2: Retrieval (Semantic Filtering)
        # We query the newly populated index using the user's original overarching query.
        # We request the Top-5 most semantically relevant documents.
        retrieved_docs = search_similar(
            query=query, 
            collection_name="deeptrace_research", 
            limit=5
        )
        
    except Exception as e:
        error_msg = f"Retriever Node failed during semantic search phase: {str(e)}"
        accumulated_errors.append(error_msg)
        print(error_msg)
        # We return an empty list so the Writer Node downstream doesn't crash,
        # but the workflow is aware that retrieval failed.
        return {
            "retrieved_documents": [],
            "errors": accumulated_errors,
            "status": "retrieval_failed"
        }

    # Success Path
    return {
        "retrieved_documents": retrieved_docs,
        "errors": accumulated_errors,
        "status": "retrieval_complete"
    }

# ==========================================
# CONNECTING THE DOTS: DATA FLOW
# ==========================================
# How this node sits in the pipeline:
# 
# 1. Search Agent -> Scrapes web -> Outputs raw `search_results`.
# 2. Retriever Agent (Here) -> Takes `search_results` -> Embeds and stores in Qdrant.
# 3. Retriever Agent (Here) -> Takes user `query` -> Asks Qdrant for Top-5 closest matches.
# 4. Retriever Agent (Here) -> Outputs highly filtered `retrieved_documents`.
# 5. Writer Agent -> Ignores `search_results` -> Reads ONLY `retrieved_documents` 
#    to write the final, hallucination-free report.

# ==========================================
# EXAMPLE INPUT / OUTPUT
# ==========================================
# Input State:
# {
#     "query": "What are the latest breakthroughs in Quantum Error Correction?",
#     "search_results": [
#         {"snippet": "Quantum computers need cold temps...", "title": "Hardware limits"},
#         {"snippet": "Recent breakthroughs in 2024 show that logical qubits...", "title": "IBM News"},
#         {"snippet": "Google stock drops...", "title": "Finance"}
#     ]
# }
#
# Expected Return Dictionary:
# {
#     "retrieved_documents": [
#         # Qdrant drops the finance and temp articles because their semantic 
#         # meaning is too far from "Error Correction". It only returns the relevant hit.
#         {"snippet": "Recent breakthroughs in 2024 show that logical qubits...", "title": "IBM News"}
#     ],
#     "errors": [],
#     "status": "retrieval_complete"
# }
