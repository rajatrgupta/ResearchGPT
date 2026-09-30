# DeepTrace Phase 3B: LLM-Powered Query Optimization
## Permanent Project Memory

This document serves as the exhaustive, definitive reference for Phase 3B of the DeepTrace project. It is designed to act as permanent project memory. Assuming it is read one year from now by a new engineer or AI system, this document contains every technical detail, architectural rationale, and state mutation required to understand, debug, or completely rebuild the Phase 3B implementation from scratch.

---

## 1. Executive Summary

### What Phase 3B Introduced
Phase 3B replaced the deterministic, regex-based Python keyword extraction system inside the retry loop with a dedicated, intelligent LangGraph node: the **Query Optimizer Agent**. 

### Why It Was Needed
In Phase 3A, the system successfully achieved autonomous self-correction by introducing the Critic agent. However, the mechanism for generating *new* search queries during a retry loop was primitive. The Search agent simply stripped "filler words" from the Critic's feedback and concatenated the remaining nouns to the original user query. This resulted in highly rigid, robotic search strings that failed to uncover nuanced data.

### What Problem It Solved
Search engines like DuckDuckGo require semantic, well-phrased queries to return high-quality results. By inserting an LLM-powered Query Optimizer between the Critic's rejection and the Search agent's next execution, DeepTrace can now "think" like a senior research librarian. It reads the specific gaps identified by the Critic and crafts highly targeted, logically distinct search strings designed explicitly to plug those gaps.

---

## 2. Problems Observed After Phase 3A

### Runtime Results & Critic Scores
During runtime validation of Phase 3A, we observed that while the Critic effectively identified missing information, the subsequent search loops often failed to raise the quality score above the `0.70` approval threshold.
*   **Cycle 1 Score:** 0.60
*   **Cycle 2 Score:** 0.62
*   **Cycle 3 Score:** 0.62 (Triggering the `MAX_ITERATIONS` safety valve).

### Why Retries Were Ineffective
The deterministic python logic was too literal. 
*   **Original Topic:** "Future of Artificial Intelligence"
*   **Critic Feedback:** "The current documents lack coverage on the economic impact regarding global supply chains."
*   **Phase 3A Generated Query:** `Future of Artificial Intelligence economic impact global supply chains`

DuckDuckGo interpreted this massive keyword string as a hyper-specific match requirement, often returning low-quality SEO spam or returning zero results and falling back to broad, unhelpful articles.

### Root Cause Analysis
The Search Agent was violating the Single Responsibility Principle. It was simultaneously trying to act as an NLP parser (extracting keywords) and an execution engine (scraping DuckDuckGo). The system lacked a cognitive layer dedicated solely to the *strategy* of information retrieval.

---

## 3. Architecture Before Phase 3B

### Diagram (Phase 3A)
```text
[START]
   ↓
[Planner]
   ↓
[Search] <─────────────────────────────┐
   │  (Runs Regex on feedback          │
   ↓   to create own retry queries)    │
[Retriever]                            │
   ↓                                   │
[Critic]                               │
   │                                   │
   ├── [Decision: RETRY] ──────────────┘
   │
   └── [Decision: APPROVED] ─→ [Writer] ─→ [END]
```

### Data Flow
1. Critic outputs string: `critic_feedback`.
2. Router reads `is_valid == False`, points back to `Search`.
3. `Search` node checks if `iteration_count > 0`.
4. If true, `Search` runs `extract_keywords(state["critic_feedback"])`.
5. `Search` appends keywords to `state["query"]`.
6. `Search` executes DuckDuckGo.

### Limitations
*   **Loss of Context:** The regex dropped verbs and adjectives that carried critical semantic weight.
*   **No Diversity:** It only generated a single concatenated string, failing to attack the missing information from multiple angles.
*   **Brittle:** If the Critic outputted an unusual sentence structure, the regex failed, generating nonsense search queries.

---

## 4. Architecture After Phase 3B

### Diagram (Phase 3B)
```text
[START]
   ↓
[Planner]
   ↓
[Search] <────────────────────────────────────────┐
   ↓                                              │ 
[Retriever]                                       │
   ↓                                              │
[Critic]                                          │
   │                                              │
   ├── [Decision: RETRY] ─→ [Query Optimizer] ────┘
   │                        (LLM generates new 
   │                         targeted queries)
   └── [Decision: APPROVED] ─→ [Writer] ─→ [END]
```

### Data Flow
1. Critic outputs string: `critic_feedback`.
2. Router reads `is_valid == False`, points to `Query Optimizer` node.
3. `Query Optimizer` receives `query` and `critic_feedback`.
4. `Query Optimizer` prompts LLM to generate exactly 3 distinct, human-like search queries.
5. LLM outputs structured JSON.
6. Graph flows unconditionally from `Query Optimizer` to `Search`.
7. `Search` iterates over `state["retry_queries"]` and executes DuckDuckGo.

---

## 5. Query Optimizer Design

### Inputs
The Query Optimizer agent reads the following from the global `ResearchState`:
*   `query` (The user's overarching topic)
*   `critic_feedback` (The exact gaps identified by the Critic)
*   `sub_questions` (The original questions, used as context to avoid searching for the exact same things again).

### Outputs
*   `retry_queries`: A list of strings.

### Prompt Strategy
The system prompt places the LLM in the persona of an "Expert Research Librarian." It is explicitly instructed:
1. Do not repeat previous searches.
2. Formulate natural language search queries optimized for a search engine like DuckDuckGo.
3. Attack the `critic_feedback` from multiple orthogonal angles.

### Structured Output Schema
To ensure reliable integration with Python logic, we utilize LangChain's `.with_structured_output()` tied to a strict Pydantic model:
```python
from pydantic import BaseModel, Field
from typing import List

class OptimizedQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 1 to 5 optimized search queries based on the feedback.",
        min_items=1,
        max_items=5
    )
```

### Pydantic Validation
The Pydantic constraints (`min_items=1`, `max_items=5`) act as a runtime barrier. If the LLM hallucinates an empty list or tries to execute 50 searches, Pydantic immediately throws a `ValidationError`, triggering the fallback logic.

### Fallback Behavior
In the event of an OpenRouter API timeout, a rate limit, or a Pydantic validation failure, the node falls back to a fail-safe mechanism: it returns a single query combining the main `query` with a truncated version of the `critic_feedback`. This prevents the graph from crashing and ensures the Search node has *something* to execute.

---

## 6. Every File Modified

### 1. `app/agents/query_optimizer.py` (NEW)
*   **Purpose:** Defines the `query_optimizer_node` function.
*   **Changes Made:** Created entirely from scratch. Implements the `OptimizedQueries` Pydantic model and the LLM invocation chain.
*   **State Fields Read:** `query`, `critic_feedback`, `sub_questions`.
*   **State Fields Written:** `retry_queries`.

### 2. `app/graph/state.py`
*   **Purpose:** Defines the `ResearchState` TypedDict memory schema.
*   **Changes Made:** Added `retry_queries: List[str]`.
*   **Why:** The `Search` node needed a place in global memory to look for the newly generated queries.

### 3. `app/graph/workflow.py`
*   **Purpose:** Compiles the LangGraph DAG.
*   **Changes Made:** 
    *   Imported `query_optimizer_node`.
    *   Added the node: `builder.add_node("query_optimizer", query_optimizer_node)`.
    *   Modified `route_after_critic`: If `is_valid == False`, return `"query_optimizer"` instead of `"search"`.
    *   Added an unconditional edge: `builder.add_edge("query_optimizer", "search")`.
*   **Why:** To seamlessly insert the new agent into the retry cycle without breaking existing flow.

### 4. `app/agents/search.py`
*   **Purpose:** Executes web searches.
*   **Changes Made:** Removed the custom Python regex/keyword extraction logic. Updated the execution loop to check `state.get("retry_queries")`. If present and `iteration_count > 0`, it loops over `retry_queries` instead of `sub_questions`.
*   **Why:** To transfer cognitive responsibility away from the Search node and strictly adhere to the list provided by the new Optimizer node.

### 5. `main.py`
*   **Purpose:** Entry point and observability.
*   **Changes Made:** Updated the initial state definition to include `"retry_queries": []`. Added logging to intercept the `query_optimizer` node stream and print: `[QueryOptimizer] Generated X targeted queries.`

---

## 7. State Evolution

### New Fields Added
*   **`retry_queries`**: `List[str]` - Holds the exact search strings to be executed on cycles 2+.

### Full `retry_queries` Lifecycle
1.  **Initialization (`main.py`):** Starts as `[]`.
2.  **Cycle 1:** Ignored by all agents.
3.  **Critic Rejection:** Router points to Query Optimizer.
4.  **Query Optimizer Execution:** LLM generates `["AI supply chain impact 2026", "Global economic disruption artificial intelligence"]`. The state key `retry_queries` is overwritten with this new list.
5.  **Search Retry:** `Search` node reads the list, executes 2 searches, and scrapes the web.
6.  **Subsequent Cycles:** If rejected again, Query Optimizer completely overwrites `retry_queries` with new angles.

---

## 8. Runtime Validation Results

During the Phase 3B validation run, the query *"Future of Artificial Intelligence"* yielded significantly superior results.

*   **Cycle 1:**
    *   *Search:* Executed 5 base questions.
    *   *Retrieval:* Pulled Top 5.
    *   *Critic Score:* `0.6` (Feedback: "Missing regulatory frameworks and international policy.")
    *   *Router Decision:* `RETRY -> Routing to Query Optimizer.`
*   **Query Optimizer Execution:**
    *   *LLM Output:* Generated 3 queries:
        1. `"Artificial Intelligence international regulatory frameworks"`
        2. `"Global AI policy and legal restrictions 2026"`
        3. `"EU AI Act and global intelligence regulation"`
*   **Cycle 2:**
    *   *Search:* Executed the 3 new targeted queries.
    *   *Retrieval:* Embedded new results alongside old, pulled the new Top 5 context chunks.
    *   *Critic Score:* `0.85` (Feedback: "Comprehensive coverage achieved.")
    *   *Router Decision:* `APPROVED -> Moving to Writer.`
*   **Final Outcome:** 
    *   The loop successfully terminated in 2 cycles instead of hitting the 3-cycle `MAX_ITERATIONS` limit. The resulting report contained explicit, highly cited paragraphs on the EU AI Act.

---

## 9. Lessons Learned

### What Worked
*   **Separation of Concerns:** Moving query generation out of the Search node massively simplified the Search node's code, making it purely a tool executor.
*   **Pydantic Enforcement:** Forcing the LLM to return `List[str]` directly into a Pydantic model completely eliminated the string parsing errors we saw in Phase 3A.

### What Failed (Initially)
*   Initially, the Query Optimizer prompt did not include `sub_questions` (the previous searches). As a result, the Optimizer occasionally regenerated the exact same search query the Planner had generated in Cycle 1, wasting a retry loop.

### Unexpected Findings
*   The system became slightly slower (adding an extra LLM call per retry loop adds ~2 seconds of latency), but this was heavily offset by the fact that the system required fewer overall retry cycles to achieve a passing score.

---

## 10. Remaining Weaknesses

Even with highly optimized queries, the system is fundamentally limited by the raw text it scrapes.

*   **Domain Authority:** DuckDuckGo might return a beautifully optimized search result from a random Reddit thread. Qdrant will blindly index it based on semantic similarity. The Writer will then treat that Reddit thread with the same factual weight as a peer-reviewed journal.
*   **Search Quality:** We are entirely reliant on free DuckDuckGo search. A commercial API (like Tavily or Google Programmable Search) would yield cleaner data.
*   **Critic Token Limit Issues:** If Qdrant retrieves 5 massive text chunks, the Critic's prompt becomes incredibly long, sometimes approaching API context window limits for smaller fallback LLMs.
*   **Duplicate Sources:** While we deduplicate URLs globally, DuckDuckGo sometimes returns the exact same article syndicated on two different URLs, leading to redundant context in Qdrant.

---

## 11. Recommended Next Phase

**Phase 4: Source Ranking & Domain Authority Scoring**

**Why retrieval quality is the next bottleneck rather than routing:**
The routing logic is now structurally sound. The system can evaluate itself and intelligently formulate strategies to find missing data. However, the *authenticity* of that data is unverified. 

To achieve a production-grade research assistant, the system must differentiate between authoritative sources (`nature.com`, `wsj.com`, `.gov`) and unreliable sources. Phase 4 will modify the Retriever to multiply the mathematical cosine similarity score by a Domain Authority weight, ensuring that when the Critic reads the "Top 5" chunks, it is reading the *highest quality* chunks, not just the most semantically adjacent ones.

---

## 12. Interview Preparation

**Q1. What is the role of the Query Optimizer in DeepTrace?**
A: It acts as an intelligent translator between the Critic's rejection and the Search agent's retry. Instead of blindly appending keywords, it uses an LLM to read the Critic's feedback and formulate highly targeted, human-like search engine queries to find the missing information.

**Q2. Why not just let the Search Agent generate its own queries?**
A: Single Responsibility Principle. The Search Agent's job is executing web scraping, handling timeouts, and managing raw data. Giving it cognitive, LLM-based query generation responsibilities makes the node bloated and harder to test.

**Q3. How does Pydantic ensure stability in the Query Optimizer?**
A: LLMs are non-deterministic. By binding the LLM's output to a Pydantic model (`OptimizedQueries`), we force it to return a structured JSON list of strings. If the LLM tries to output conversational text, Pydantic throws a validation error, preventing the pipeline from crashing further down.

**Q4. Explain LangGraph Conditional Routing in the context of Phase 3B.**
A: After the Critic scores the data, a Python function `route_after_critic` evaluates the state. If the score is `< 0.70`, the router returns `"query_optimizer"`. This creates a cyclic state machine. The graph will flow: `Critic -> Router -> Query Optimizer -> Search -> Retriever -> Critic`, repeating until the score passes or `MAX_ITERATIONS` is reached.

**Q5. What happens if the Query Optimizer's LLM call fails?**
A: It utilizes a "Fail-Safe" block. It catches the exception and deterministically returns a single basic query combining the original overarching topic with a snippet of the Critic's feedback, ensuring the Search node has something to execute rather than crashing the application.

**Q6. What is RAG and how does Phase 3B improve it?**
A: RAG (Retrieval-Augmented Generation) prevents hallucinations by forcing an LLM to answer only based on retrieved database facts. Phase 3B improves our RAG pipeline by ensuring that if the retrieved facts are insufficient, the system knows exactly *how* to ask the internet for better facts before writing the final report.

**Q7. Why did Phase 3A's deterministic keyword appending fail?**
A: Search engines rely on semantic relationships. Appending raw nouns (e.g., "AI economic supply chain global") often breaks the search engine's intent parsing, resulting in SEO spam. A natural language query ("How does AI impact global supply chains economically") yields vastly better journalistic results.

---

## 13. Development Timeline

*   **Phase 1:** Core Linear Pipeline (Planner -> Search -> Writer).
*   **Phase 2:** Persistent Memory & RAG (Integration of Qdrant & Retriever).
*   **Phase 3A:** Evaluation & Conditional Routing (Critic Agent & deterministic regex retry loops).
*   **Phase 3B:** LLM-Powered Query Optimization (Query Optimizer Agent replaces deterministic logic).
*   **Phase 4:** *(Planned)* Source Ranking & Domain Authority.
*   **Phase 5:** *(Planned)* Production UI & Cloud Deployment.