"""
Critic Agent for DeepTrace

This module defines the Critic Node. Its responsibility is to act as a 
Quality Assurance (QA) gatekeeper. It evaluates the semantically filtered 
retrieved documents against the user's original query and outputs a score 
and feedback.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from app.graph.state import ResearchState
from app.core.llm import get_llm

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
# Deterministic threshold for routing. 
# By defining this in Python rather than prompt engineering the LLM to guess 
# validity, we gain strict, observable control over the system's strictness.
QUALITY_THRESHOLD = 0.70

# ==========================================
# PYDANTIC SCHEMA
# ==========================================
# We define a strict schema to force the LLM to return a predictable JSON object.
# This prevents the LLM from outputting conversational filler and ensures our
# Python code can reliably parse the 'quality_score' as a float.
class CriticEvaluation(BaseModel):
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0 indicating how well the retrieved context answers the user query. 1.0 is a perfect, comprehensive answer. 0.0 is completely irrelevant or missing data."
    )
    critic_feedback: str = Field(
        description="A single, concise sentence explaining the score. If the score is low, state exactly what information is missing. Example: 'Missing specific statistics and recent dates.'"
    )

def format_retrieved_documents_for_eval(documents: list[Dict[str, Any]]) -> str:
    """
    Helper function to format documents for the Critic's prompt, 
    including rich metadata for better evaluation.
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

def critic_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the evaluation phase of the research workflow.
    
    Args:
        state: The current ResearchState containing 'query' and 'retrieved_documents'.
        
    Returns:
        A dictionary containing 'quality_score', 'critic_feedback', 'is_valid',
        and a 'status' indicator.
    """
    
    query = state.get("query", "")
    retrieved_documents = state.get("retrieved_documents", [])
    
    # Defensive programming: If we have no documents, the score is automatically 0.
    if not retrieved_documents:
        return {
            "quality_score": 0.0,
            "critic_feedback": "No documents retrieved. Complete failure.",
            "is_valid": False,
            "status": "validation_failed"
        }

    # Format the rich context for the LLM
    context_str = format_retrieved_documents_for_eval(retrieved_documents)

    # Prompt Engineering: We frame the LLM as a harsh grading system.
    # We strictly enforce the 0.0 - 1.0 range in the prompt.
    prompt = PromptTemplate(
        template="""You are an expert Research Editor. Your job is to evaluate if the provided retrieved documents contain enough factual information to write a comprehensive, high-quality report answering the user's query.

User Query: {query}

--- RETRIEVED DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Evaluate the coverage, relevance, and completeness of the documents against the query.
Assign a score between 0.0 (useless) and 1.0 (perfect). 
YOU MUST OUTPUT A SCORE BETWEEN 0.0 AND 1.0.

Be harsh. If critical sub-topics are missing, lower the score.
""",
        input_variables=["query", "context"]
    )

    # Initialize the centralized LLM
    llm_config = get_llm()
    robust_llm = llm_config["llm"]

    # RAG UPGRADE: We bind our Pydantic schema to the LLM. 
    # This forces OpenRouter models to return structured JSON matching CriticEvaluation.
    structured_llm = robust_llm.with_structured_output(CriticEvaluation)
    
    chain = prompt | structured_llm

    try:
        # Execute the evaluation chain
        result: CriticEvaluation = chain.invoke({
            "query": query,
            "context": context_str
        })
        
        # DETERMINISTIC DECISION LOGIC
        # We decouple evaluation from routing. The LLM only evaluates (scores).
        # The Python code makes the hard deterministic routing decision.
        is_valid = result.quality_score >= QUALITY_THRESHOLD
        
        status = "validation_passed" if is_valid else "validation_failed"
        
        return {
            "quality_score": result.quality_score,
            "critic_feedback": result.critic_feedback,
            "is_valid": is_valid,
            "status": status
        }
        
    except Exception as e:
        error_msg = f"Critic failed during evaluation: {str(e)}"
        print(f"Warning: {error_msg}")
        # FAIL-CLOSED LOGIC:
        # If the LLM call fails (e.g., API outage), we do NOT approve the data.
        # We fail closed, forcing a retry or a graceful exit if MAX_ITERATIONS is reached.
        return {
            "errors": [error_msg],
            "quality_score": 0.0,
            "critic_feedback": error_msg,
            "is_valid": False, 
            "status": "validation_error"
        }
