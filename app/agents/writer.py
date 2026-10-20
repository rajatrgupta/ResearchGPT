"""
Writer Agent for DeepTrace
"""

import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from app.graph.state import ResearchState
from app.core.llm import get_llm

class Citation(BaseModel):
    claim: str = Field(description="The factual claim")
    source_url: str = Field(description="URL of the retrieved document")

class ReportOutput(BaseModel):
    report_content: str = Field(description="The main markdown report. MUST embed inline citations strictly as [Source: URL]")
    citations: List[Citation] = Field(description="List of all citations used")

def format_retrieved_documents(documents: list[Dict[str, Any]]) -> str:
    formatted_context = ""
    sources_list = []
    for i, doc in enumerate(documents):
        question = doc.get("question", "General Topic")
        title = doc.get("title", "Untitled")
        snippet = doc.get("snippet", "")
        source = doc.get("source") or doc.get("link") or "No Source"
        
        formatted_context += f"--- Source [{i+1}] ---\n"
        formatted_context += f"Related Sub-Question: {question}\n"
        formatted_context += f"Title: {title}\n"
        formatted_context += f"Snippet: {snippet}\n"
        formatted_context += f"Link: {source}\n\n"
        
        if source and source != "No Source" and source != "error" and source != "timeout":
            sources_list.append(f"- [{title}]({source})")
            
    if sources_list:
        formatted_context += "--- AVAILABLE SOURCES FOR INLINE CITATIONS ---\n"
        formatted_context += "Use these exact Markdown links when citing facts:\n"
        formatted_context += "\n".join(sources_list) + "\n\n"
        
    return formatted_context

def writer_node(state: ResearchState) -> Dict[str, Any]:
    query = state.get("query", "")
    retrieved_documents = state.get("retrieved_documents", [])
    degradation_mode = state.get("degradation_mode", False)
    
    if degradation_mode:
        fallback_report = f"# Research Report: {query}\n\n"
        fallback_report += "## Insufficient Data / Exhausted Research Limits\n"
        fallback_report += "The autonomous research system exhausted its maximum retry loops without finding enough high-quality context to meet the minimum verification threshold. The generated report has been aborted to prevent hallucinations."
        
        return {
            "report": fallback_report,
            "status": "writing_degraded"
        }

    if not retrieved_documents:
        fallback_report = f"# Research Report: {query}\n\n"
        fallback_report += "## Error\n"
        fallback_report += "Unfortunately, the retrieval system was unable to find highly relevant documents for this query. The report generation was aborted to prevent hallucination."
        
        return {
            "report": fallback_report,
            "status": "writing_failed",
            "errors": ["Writer received empty retrieved_documents list."]
        }

    context_str = format_retrieved_documents(retrieved_documents)

    prompt = PromptTemplate(
        template="""You are a Tier-1 Strategy Consultant and AI Researcher (like McKinsey or BCG). Your task is to synthesize the retrieved documents into a highly dense, analytical, and professional Markdown report.

STRICT RULES - READ CAREFULLY:
1. MANDATORY INLINE CITATIONS: Every key claim, metric, and bullet point MUST end with an inline Markdown hyperlink citation formatted strictly as `[[Source Name](URL)]` using the exact links provided in the findings. Uncited claims are strictly forbidden.
   Example format:
   - **Genomic Alignment:** Quantum annealing algorithms optimize multigenic alignment latency by up to 40% [[Nature](https://www.nature.com/articles/...)].
2. NO FLUFF or FILLER: Never use generic introductory/concluding sentences. Start directly with high-impact insights.
3. NO REPETITION & STRICT THEMATIC ISOLATION: Never repeat the same concept across different sections. Group related points so each section provides strictly net-new depth.
4. DEEP GRANULARITY: Do not stay at the surface level. Extract specific metrics, named institutions, algorithmic benchmarks, and historical frameworks.
5. PROFESSIONAL FORMATTING: Use H2/H3 headers, bullet points for metrics, and bold text for key terminology.
6. EXECUTIVE SYNTHESIS: You must ALWAYS conclude the report with a final section titled "## Executive Synthesis" (a brief 2-3 sentence paragraph answering the "So what?").
7. REFERENCES SECTION: Append a final "## Key Sources & References" section listing all unique source URLs cited.

Output the final Markdown report directly. Do NOT wrap it in JSON.

User Query: {query}

--- RESEARCH FINDINGS ---
{context}
--- END FINDINGS ---

Write the report now:
""",
        input_variables=["query", "context"]
    )

    llm_config = get_llm()
    writer_llm = llm_config["llm"]
    
    chain = prompt | writer_llm

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            if attempt == 1:
                print("[Writer] START LLM report synthesis...")
            else:
                print(f"[Writer] RETRY {attempt}/{max_retries} LLM report synthesis...")
            
            result = chain.invoke({
                "query": query,
                "context": context_str
            })
            
            raw_content = result.content
            if isinstance(raw_content, list):
                final_content = "\n".join([chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw_content])
            elif not isinstance(raw_content, str):
                final_content = str(raw_content)
            else:
                final_content = raw_content

            elapsed = time.time() - start_time
            print(f"[Writer] COMPLETE LLM report synthesis in {elapsed:.2f}s")
            
            return {
                "report": final_content,
                "status": "writing_complete"
            }
            
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = "503" in error_str or "unavailable" in error_str or "connection" in error_str or "timeout" in error_str or "429" in error_str or "exhausted" in error_str
            
            if is_retryable and attempt < max_retries:
                wait_time = 10 * attempt # progressive backoff for rate limits
                print(f"[Writer] Warning: LLM call failed with retryable error (attempt {attempt}/{max_retries}). Waiting {wait_time}s... Error: {str(e)}")
                time.sleep(wait_time)
                continue
                
            elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
            error_msg = f"Writer failed to generate report in {elapsed:.2f}s: {str(e)}"
            print(f"[Writer] FAILED LLM report synthesis after {attempt} attempts: {error_msg}")
            return {
                "errors": [error_msg],
                "status": "writing_failed",
                "report": "An error occurred during report generation."
            }
