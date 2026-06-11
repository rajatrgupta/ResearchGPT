"""
Writer Agent for DeepTrace

This module defines the Writer Node. Its responsibility is to take the semantically
filtered retrieved documents and synthesize them into a cohesive, well-formatted 
markdown report with citations.
"""

from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from app.graph.state import ResearchState
from app.core.llm import get_llm

# ==========================================
# RULE 1: WHAT IS THIS CONCEPT?
# ==========================================
# The Writer Agent executes the Generation phase of RAG (Retrieval-Augmented Generation).
# 
# Why context quality over quantity?
# In Phase 1, we fed the LLM ALL raw search results. This risks confusing the LLM 
# with noise, exceeding token limits, and triggering the "Lost in the Middle" problem 
# where LLMs ignore data buried in large prompts.
#
# Now, we strictly use `retrieved_documents`. By feeding the LLM a much smaller, 
# mathematically curated list of highly relevant facts, we drastically reduce 
# hallucinations and ensure a grounded, accurate synthesis.

def format_retrieved_documents(documents: list[Dict[str, Any]]) -> str:
    """
    Helper function to convert the semantically filtered documents into a 
    readable string for the LLM prompt context.
    """
    formatted_context = ""
    for i, doc in enumerate(documents):
        # We handle cases where the payload might be missing keys
        question = doc.get("question", "General Topic")
        title = doc.get("title", "Untitled")
        snippet = doc.get("snippet", "")
        source = doc.get("source", "No Source")
        
        formatted_context += f"--- Source [{i+1}] ---\n"
        formatted_context += f"Related Sub-Question: {question}\n"
        formatted_context += f"Title: {title}\n"
        formatted_context += f"Snippet: {snippet}\n"
        formatted_context += f"Link: {source}\n\n"
        
    return formatted_context

def writer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the synthesis and reporting phase of the RAG workflow.
    
    Args:
        state: The current ResearchState containing 'query' and 'retrieved_documents'.
        
    Returns:
        A dictionary containing the final synthesized 'report', 'status', 
        and optionally 'errors'.
    """
    
    query = state.get("query", "")
    
    # RAG UPGRADE: We explicitly stop using raw 'search_results'.
    # We now pull strictly from the highly curated 'retrieved_documents'.
    retrieved_documents = state.get("retrieved_documents", [])
    
    # Defensive programming: Handle empty retrieval gracefully.
    # If Qdrant failed or returned nothing, we cannot write a grounded report.
    if not retrieved_documents:
        fallback_report = f"# Research Report: {query}\n\n"
        fallback_report += "## Error\n"
        fallback_report += "Unfortunately, the retrieval system was unable to find "
        fallback_report += "highly relevant documents for this query. The report generation was aborted to prevent hallucination."
        
        return {
            "report": fallback_report,
            "status": "writing_failed",
            "errors": ["Writer received empty retrieved_documents list."]
        }

    # Format the high-quality context for the LLM
    context_str = format_retrieved_documents(retrieved_documents)

    # Prompt Engineering: Force specific markdown sections and inline citations.
    # We heavily instruct the LLM to rely ONLY on the provided context.
    prompt = PromptTemplate(
        template="""You are an expert research analyst and technical writer.
Your task is to synthesize the provided research findings into a highly cohesive, professional markdown report answering the user's original query.

User Query: {query}

You MUST structure your report EXACTLY with the following markdown headers:
# Introduction
# Key Findings
# Analysis
# Conclusion

Rules:
1. Synthesize the information based ONLY on the provided RESEARCH FINDINGS. Do not invent or hallucinate facts.
2. Include inline citations where possible using the exact format: [Source: https://...] based on the provided Links.
3. Ensure smooth transitions between sections.

--- RESEARCH FINDINGS ---
{context}
--- END FINDINGS ---

Write the report now:
""",
        input_variables=["query", "context"]
    )

    # Initialize the centralized, fault-tolerant LLM
    llm_config = get_llm()
    robust_llm = llm_config["llm"]

    # Create the execution chain
    chain = prompt | robust_llm

    try:
        # Execute the chain
        result = chain.invoke({
            "query": query,
            "context": context_str
        })
        
        # Extract string content from the AIMessage
        report_content = result.content
        
        return {
            "report": report_content,
            "status": "writing_complete"
        }
        
    except Exception as e:
        error_msg = f"Writer failed to generate report: {str(e)}"
        return {
            "errors": [error_msg],
            "status": "writing_failed",
            "report": "An error occurred during report generation."
        }

# ==========================================
# CONNECTING THE DOTS
# ==========================================
# How this writer connects to the ecosystem:
# 
# - Retriever Agent: The Retriever used Qdrant to filter out the noise from the raw search. 
#   The Writer relies entirely on the 'snippet' and 'source' fields provided by the 
#   Retriever in `retrieved_documents`.
# 
# - Future Critic Agent: The Critic will review THIS generated report against the original 
#   query and the `retrieved_documents` to ensure the Writer didn't hallucinate facts 
#   outside of the provided context.
# 
# - Future PostgreSQL storage: The output of this agent (`report`) is the final 
#   artifact that will be served to the Streamlit UI and stored long-term.

# ==========================================
# EXAMPLE INPUT / OUTPUT
# ==========================================
# Input State:
# {
#     "query": "Quantum Computing Timelines",
#     "retrieved_documents": [
#         {"question": "Hardware?", "title": "IBM News", "snippet": "IBM says 2029...", "source": "https://ibm.com"}
#     ]
# }
#
# Expected Return Dictionary:
# {
#     "report": "# Introduction\nQuantum computing... \n\n# Key Findings\nIBM targets 2029 [Source: https://ibm.com]... \n\n# Analysis\n... \n\n# Conclusion\n...",
#     "status": "writing_complete"
# }
