"""
Critic Agent for DeepTrace
"""

import time
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from app.graph.state import ResearchState
from app.core.llm import get_llm

QUALITY_THRESHOLD = 0.70

class CriticEvaluation(BaseModel):
    faithfulness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0 indicating how grounded the information is in the retrieved context. Are there hallucinations?"
    )
    relevancy_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score between 0.0 and 1.0 indicating how well the retrieved context answers the original user query."
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="An overall combined score between 0.0 and 1.0. 1.0 is a perfect, comprehensive answer."
    )
    critic_feedback: str = Field(
        description="A single, concise sentence explaining the score and what information is missing."
    )

def format_retrieved_documents_for_eval(documents: list[Dict[str, Any]]) -> str:
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
    query = state.get("query", "")
    retrieved_documents = state.get("retrieved_documents", [])
    
    if not retrieved_documents:
        return {
            "quality_score": 0.0,
            "critic_feedback": "No documents retrieved. Complete failure.",
            "is_valid": False,
            "status": "validation_failed"
        }

    context_str = format_retrieved_documents_for_eval(retrieved_documents)

    prompt = PromptTemplate(
        template="""You are an elite Research Evaluator and Peer-Review Referee.
Current Reference Year: 2026.
Your goal is to evaluate whether the retrieved context contains sufficient depth, empirical data, and verifiable evidence to synthesize a publication-grade executive research report.

DYNAMIC EVALUATION CRITERIA:
- Depth & Specificity: Does the context contain hard numbers, empirical benchmarks, percentages, named clinical trials/studies, sample sizes, dollar figures, or named institutional frameworks? Penalize vague, superficial summaries that only state generic qualitative claims.
- Temporal Freshness: Does the context include up-to-date developments (2024-2026) rather than outdated speculation?
- Source Groundedness: Are the findings attributed to recognizable, trustworthy publications and direct URLs?

EVALUATION METRICS (RAGAS-Style):
1. Faithfulness: Is the information factual and free of contradictions/hallucinations based strictly on the context?
2. Relevancy: Does the provided context directly answer the User Query with technical and empirical rigor?
3. Quality Score: An overall judgment (0.0 to 1.0). Score >= 0.70 ONLY if the context has sufficient factual density to write an authoritative, non-fluffy report.
4. Actionable Feedback: If Quality Score < 0.70, provide a specific sentence stating EXACTLY what empirical metrics or dimensions are missing to guide query optimization.

Output your response strictly in the required JSON format containing the scores and actionable feedback.

User Query: {query}

--- RETRIEVED DOCUMENTS ---
{context}
--- END DOCUMENTS ---
""",
        input_variables=["query", "context"]
    )

    llm_config = get_llm(structured_schema=CriticEvaluation)
    structured_llm = llm_config["llm"]
    chain = prompt | structured_llm

    try:
        start_time = time.time()
        print("[Critic] START LLM evaluation with Ragas metrics...")
        
        result: CriticEvaluation = chain.invoke({
            "query": query,
            "context": context_str
        })
        
        elapsed = time.time() - start_time
        print(f"[Critic] COMPLETE LLM evaluation in {elapsed:.2f}s (Score: {result.quality_score:.2f})")
        
        is_valid = result.quality_score >= QUALITY_THRESHOLD
        status = "validation_passed" if is_valid else "validation_failed"
        
        return {
            "quality_score": result.quality_score,
            "critic_feedback": result.critic_feedback,
            "is_valid": is_valid,
            "status": status
        }
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
        error_msg = f"Critic failed during evaluation in {elapsed:.2f}s: {str(e)}"
        print(f"[Critic] FAILED LLM evaluation: {error_msg}")
        return {
            "errors": [error_msg],
            "quality_score": 0.0,
            "critic_feedback": error_msg,
            "is_valid": False, 
            "status": "validation_error"
        }
