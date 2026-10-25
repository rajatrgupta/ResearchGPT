"""
Planner Agent for DeepTrace

This module defines the Planner Node. Its responsibility is to take a broad
research query and break it down into exactly 5 focused, actionable sub-questions.
"""

import time
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from app.graph.state import ResearchState
from app.core.llm import get_llm

# ==========================================
# RULE 1: WHAT IS THIS CONCEPT?
# ==========================================
# The Planner Agent is the "architect" of our research process.
# 
# Why planning improves retrieval quality:
# Without planning:
# User Query -> One broad search -> Mixed results -> Lower report quality
# 
# With planning:
# User Query -> 5 focused sub-questions -> 5 focused searches -> 
# Better coverage -> Better retrieval quality -> Better final report
#
# We mandate that the LLM must "Return JSON format from LLM". We enforce 
# and validate this using "Pydantic output parsing".

# 1. Define the desired output structure using Pydantic
class SubQuestions(BaseModel):
    """Schema for the expected output from the Planner LLM."""
    questions: list[str] = Field(
        description="Exactly 5 research-focused sub-questions derived from the main query."
    )

def planner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executes the planning phase of the research workflow.
    
    Args:
        state: The current ResearchState containing the original 'query'.
        
    Returns:
        A dictionary containing the generated 'sub_questions', 'status', 
        and optionally 'errors' to be merged into the main state.
    """
    
    query = state.get("query", "")
    
    # Defensive check: if there's no query, we can't plan.
    if not query:
        return {
            "errors": ["Planner received an empty query."],
            "status": "planning_failed"
        }

    prompt = PromptTemplate(
        template="""You are a Principal Research Director and Lead Technical Analyst.
Current Reference Year: 2026.
Your task is to break down the user's research query into EXACTLY 5 distinct, empirically focused, and technical sub-questions.

CRITICAL INSTRUCTIONS FOR HIGH-QUALITY RESEARCH:
1. Avoid vague, philosophical, or broad conversational questions (e.g., do NOT ask "What are the ethical implications?").
2. Formulate each sub-question to target EMPIRICAL DATA, BENCHMARKS, SPECIFIC ARCHITECTURES, RECENT STUDIES (2024-2026), AND QUANTITATIVE METRICS.
3. Include high-yield search keywords where relevant, such as: "benchmarks", "sensitivity specificity accuracy", "clinical trials results", "ROI cost analysis", "FDA EMA regulatory clearance", "latency throughput parameters", "market adoption CAGR USD".
4. Ensure the 5 questions cover orthogonal dimensions:
   - Dimension 1: Technical & Algorithmic Architecture / Core Mechanisms
   - Dimension 2: Empirical Benchmark Comparisons & Quantitative Performance Metrics
   - Dimension 3: Real-World Clinical / Industrial Deployment & Operational Case Studies
   - Dimension 4: Economic Costs, Unit Economics, ROI & Market Dynamics
   - Dimension 5: Regulatory Compliance, Safety Standards & Future 2026-2030 Roadmap

User Query: {query}
""",
        input_variables=["query"]
    )

    # 4. Initialize the LLM using our centralized, fault-tolerant structured configuration
    llm_config = get_llm(structured_schema=SubQuestions)
    structured_llm = llm_config["llm"]

    # 5. Create the execution chain
    chain = prompt | structured_llm

    try:
        # Execute the chain
        start_time = time.time()
        print("[Planner] START LLM generation...")
        
        result: SubQuestions = chain.invoke({"query": query})
        
        elapsed = time.time() - start_time
        print(f"[Planner] COMPLETE LLM generation in {elapsed:.2f}s")
        
        questions = result.questions if hasattr(result, "questions") else []
        if len(questions) > 5:
            questions = questions[:5]
        elif len(questions) == 0:
            raise ValueError("Planner returned an empty list of questions.")
        
        return {
            "sub_questions": questions,
            "status": "planning_complete"
        }
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
        error_msg = f"Planner failed in {elapsed:.2f}s: {str(e)}"
        print(f"[Planner] FAILED LLM generation: {error_msg}")
        
        # High-quality fallback sub-questions so temporary provider hiccups never kill research
        fallback_questions = [
            f"{query}: Core technical architecture, algorithmic mechanisms, and specifications",
            f"{query}: Empirical benchmark performance, accuracy, sensitivity, and quantitative metrics",
            f"{query}: Real-world clinical industrial deployment, trials, and operational outcomes",
            f"{query}: Economic evaluation, cost per unit, ROI, and commercial market adoption CAGR",
            f"{query}: Regulatory frameworks FDA EMA, safety standards, and 2026-2030 strategic outlook"
        ]
        return {
            "errors": [error_msg],
            "status": "planning_complete",
            "sub_questions": fallback_questions
        }

# ==========================================
# CONNECTING THE DOTS
# ==========================================
# How this planner connects to the ecosystem:
# - LangGraph State: It reads state["query"] and updates state["sub_questions"] 
#   and state["status"].
# - Search Agent: The Search Agent will loop through the generated state["sub_questions"]
#   and perform a distinct web search for each one.
# - Future Critic Agent: If the Search Agent's findings are poor, the Critic 
#   will review these original sub_questions to see if the plan itself was flawed.

# ==========================================
# EXAMPLE INPUT / OUTPUT
# ==========================================
# Input State:
# {"query": "The impact of artificial intelligence on modern healthcare"}
#
# Expected Return Dictionary:
# {
#     "sub_questions": [
#         "How are AI algorithms currently being utilized in diagnostic imaging and early disease detection?",
#         "What are the ethical and privacy concerns surrounding patient data in AI-driven healthcare systems?",
#         "In what ways is artificial intelligence accelerating pharmaceutical research and drug discovery?",
#         "How is AI being used to personalize treatment plans and patient care pathways?",
#         "What are the economic impacts and cost-saving potentials of integrating AI into hospital administration?"
#     ],
#     "status": "planning_complete"
# }
