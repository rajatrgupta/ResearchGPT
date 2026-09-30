# DeepTrace Phase 3: Autonomous Evaluation and Conditional Routing
## Permanent Project Memory

This document serves as the definitive reference for Phase 3 of the DeepTrace project. It is designed to preserve deep context for returning engineers, subsequent AI agents, or as preparation for technical interviews. It thoroughly details the introduction of the Critic Agent, Conditional Routing, the Self-Correcting Iterative Research Loop, and future roadmaps.

---

## SECTION 1 — PROJECT OVERVIEW
═══════════════════════════════════════

### Project Name
**DeepTrace**

### Project Goal
To build a production-grade, fully autonomous Multi-Agent Research System using LangGraph. Given a user query, DeepTrace autonomously plans sub-queries, executes robust web searches, indexes findings in a local vector database, evaluates its own research quality, self-corrects via iterative search loops, and synthesizes hallucination-free, heavily cited markdown reports.

### Current Architecture After Phase 3
Phase 3 transformed DeepTrace from a linear pipeline into a **cyclic, self-correcting state machine**. By introducing a Critic Agent and Conditional Routing, the system can now judge its own work and loop backward to gather more data if its quality standards are not met.

### End-to-End Workflow Diagram

```text
[START]
   ↓
[Planner]     <-- Deconstructs user query into targeted sub-questions
   ↓
[Search] <─────────────────────────────┐
   ↓                                   │ (Retry Path: Uses Critic Feedback)
[Retriever]                            │
   ↓                                   │
[Critic]      <-- QA Gatekeeper        │
   │                                   │
   ├────── (Evaluate Quality)          │
   │                                   │
   ├── [Decision: RETRY] ──────────────┘
   │
   └── [Decision: APPROVED] ───────────┐
                                       ↓
                                    [Writer] <-- Synthesizes final report
                                       ↓
                                     [END]
```

---

## SECTION 2 — WHY PHASE 3 WAS NEEDED
═══════════════════════════════════════

### Problems with Phase 2

While Phase 2 successfully introduced RAG (Retrieval-Augmented Generation) and persistent vector memory via Qdrant, it suffered from a fundamental flaw inherent to **linear workflows**: it assumed the first attempt was always successful.

**1. No Quality Checking & No Self-Correction:**
In Phase 2, the pipeline flowed exactly once: `Planner -> Search -> Retriever -> Writer`. If the Search Agent happened to scrape low-quality, tangential, or promotional SEO articles, the Retriever would blindly index them. The Writer would then faithfully write a report based on that bad data. The system had no mechanism to say, "Wait, this data isn't good enough. Let's search again."

**2. Why RAG / Retrieval Could Still Be Weak:**
RAG prevents an LLM from hallucinating facts outside of its context window, but it **does not guarantee that the context window itself contains the right answer.** If you search for "Apple" expecting fruit, but DuckDuckGo returns tech articles, Qdrant will retrieve the "most relevant" tech articles. 

**3. The Mediocre Context Problem (Real-World Example):**
*   **Query:** "Future of Artificial Intelligence"
*   **Phase 2 Execution:** The search returns 15 articles. 12 are promotional blogs about "Buy our AI stock," and 3 mention AI replacing teachers.
*   **Result:** The Retriever passes the 3 best articles to the Writer. The Writer produces a report claiming the "Future of AI is strictly limited to the education sector." 
*   **Conclusion:** The report is factually accurate to the provided context, but it is a *terrible* overall research report because it lacks depth, breadth, and critical analysis.

Phase 3 was built to solve this by installing a "Senior Editor" (The Critic) who refuses to let the Writer generate a report until the retrieved context is actually comprehensive.

---

## SECTION 3 — CRITIC AGENT
═══════════════════════════════════════

### What is a Critic Agent?
A Critic Agent is a specialized node in a multi-agent system dedicated exclusively to evaluating the quality, accuracy, and completeness of the work produced by upstream agents. It does not scrape new data or synthesize reports; its sole purpose is to judge existing data against a rubric.

### Why AI Systems Use Critics
Generative models are inherently biased toward being "helpful." If you ask an LLM to write a report using bad data, it will eagerly write a beautifully formatted, completely useless report. By separating the "creator" role from the "evaluator" role, we introduce adversarial friction that significantly elevates the final output quality.

### How Our Critic Works
The Critic Node receives the user's original `query` and the `retrieved_documents` pulled from Qdrant. It prompts an LLM to act as a harsh grading system, evaluating the coverage, relevance, and source diversity of the documents.

#### The `CriticEvaluation` Schema
To ensure the LLM doesn't output conversational filler (e.g., "Sure, I can evaluate that!"), we bind the LLM to a strict Pydantic JSON schema:
*   **`quality_score` (float):** A strict numerical value between 0.0 and 1.0. (Enforced via Pydantic `ge=0.0`, `le=1.0`).
*   **`critic_feedback` (str):** A concise sentence explaining what is missing (e.g., "Missing data on AI regulation").

#### Deterministic Validation (A Crucial Design Choice)
Initially, one might design the schema to ask the LLM to output an `is_valid` boolean directly. **We explicitly avoided this.**

**Why "LLM evaluates, Python decides" is better than "LLM evaluates, LLM decides":**
LLMs are non-deterministic. They struggle with strict threshold logic. If we rely on the LLM to output `True` or `False`, it might approve a mediocre result simply because of prompt variation. 

Instead, the LLM *only* provides the numerical score. Our Python code executes the deterministic logic:
```python
QUALITY_THRESHOLD = 0.70
is_valid = result.quality_score >= QUALITY_THRESHOLD
```
This gives us strict, observable control. If we want the system to be harsher, we simply change the Python constant to `0.85`.

---

## SECTION 4 — CONDITIONAL ROUTING
═══════════════════════════════════════

### LangGraph Conditional Edges
In a Directed Acyclic Graph (DAG), standard edges (`add_edge("A", "B")`) rigidly force execution down a single path. Conditional edges transform the graph into a dynamic state machine. Instead of a hardcoded target, a conditional edge executes a Python function (a router). This function reads the current state and dynamically returns the name of the next node to execute.

### The `route_after_critic()` Function
This router sits immediately after the Critic node and makes the final decision on graph flow based on three rules:

1.  **Approval (Score >= 0.70):** 
    If `state["is_valid"] == True`, the router returns `"writer"`. The research is deemed sufficient, and the report is synthesized.
2.  **Retry (Score < 0.70):** 
    If the score is too low, the router returns `"search"`. This forces the pipeline to loop backward, triggering a new search cycle to find better data.
3.  **Safety Valve (MAX_ITERATIONS reached):** 
    If the score is 0.45, but the system has already looped 3 times (`iteration_count >= MAX_ITERATIONS`), the router returns `"writer"`. This forces the system to synthesize the best report it can, preventing infinite loops and catastrophic API token burns.

---

## SECTION 5 — ITERATIVE RESEARCH LOOP
═══════════════════════════════════════

### The Cognitive Workflow
The cyclic architecture (`Search -> Evaluate -> Retry -> Evaluate -> Approve`) mimics how professional human researchers operate. A human rarely executes a perfect Google search on their first attempt. They search, read the results, realize they missed a critical angle or used the wrong keywords, and search again. 

### State Transitions During the Loop
1.  **Iteration 0 (Search):** Planner asks 5 questions. Search scrapes 15 links. State `search_results` grows to 15. `iteration_count` becomes 1.
2.  **Iteration 1 (Evaluate):** Retriever indexes 15 links, pulls Top 5. Critic scores them a 0.4. Router says RETRY.
3.  **Iteration 1 (Retry Search):** Search node sees `iteration_count > 0` and reads the Critic's feedback. It generates new targeted queries, scrapes 10 new links, and **appends** them to the state. `search_results` grows to 25. `iteration_count` becomes 2.
4.  **Iteration 2 (Evaluate):** Retriever indexes the *new* combined pool of 25 links, pulling the new Top 5. Critic scores them a 0.8. Router says APPROVED.

Because the state uses `operator.add` for `search_results`, data is never lost during loops; the research pool simply grows deeper and richer until the Critic is satisfied.

---

## SECTION 6 — SEARCH RETRY SYSTEM
═══════════════════════════════════════

When the Critic rejects the current retrieved data, it provides string-based `critic_feedback` detailing the gaps. Passing this feedback into the retry loop is critical, otherwise the Search Agent would blindly re-execute the original sub-questions and scrape the same missing data.

### Deterministic Keyword Extraction
Instead of injecting another LLM call to rewrite queries based on the feedback, we implemented a lightweight, deterministic Python function `generate_retry_queries()`. This approach strips editorial filler phrases (e.g., "coverage is missing") and stop words, leaving only the high-signal terms. It then appends these terms to the original query to force DuckDuckGo to search specifically for the missing angle.

### Why Avoid Another LLM Call?
*   **Cheaper:** Eliminates API costs for query reformulation.
*   **Faster:** Removes network latency from the retry loop.
*   **Deterministic:** Python string parsing always yields predictable queries, preventing the LLM from over-complicating or hallucinating search terms.
*   **Fewer API Calls:** Adheres to single-responsibility principles for the Search Node (it searches, it doesn't reason).

### Retry Query Example
*   **Original Query:** `Future of Artificial Intelligence`
*   **Critic Feedback:** `Missing regulation and ethics.`
*   **Extracted Keywords:** `["regulation", "ethics"]`
*   **Generated Queries:**
    *   `Future of Artificial Intelligence regulation`
    *   `Future of Artificial Intelligence ethics`

These generated queries directly replace the Planner's original sub-questions on iteration cycles > 0.

---

## SECTION 7 — OBSERVABILITY SYSTEM
═══════════════════════════════════════

Visibility into the execution of a multi-agent loop is essential to avoid the "black box" effect. We implemented real-time observability in `main.py`.

### `graph.invoke()` vs `graph.stream()`
*   **`invoke()`:** A synchronous call that executes the entire state machine and only returns the final state when the graph hits the END node. The user sees nothing until completion.
*   **`stream()`:** A generator that yields a dictionary containing partial state updates every time a single node completes its execution. 

### Implementation Highlights
By iterating over `graph.stream(initial_state)`, we achieved:
*   **Node Tracking:** Logging exactly when `[Search]`, `[Retriever]`, or `[Critic]` starts and finishes.
*   **Cycle Tracking:** Dynamically detecting retry loops via the `iteration_count` and printing headers like `--- CYCLE #2 ---`.
*   **Router Visibility:** Displaying the numerical quality score, the Critic's textual feedback, and explicitly logging the Router's decision (`APPROVED` or `RETRY`).
*   **State Merging:** A custom `merge_state` function was built to mimic LangGraph's internal `operator.add` reducers, allowing us to safely accumulate `search_results` and print real-time statistics (e.g., Total Docs Retrieved) while the loop ran.

---

## SECTION 8 — STATE MANAGEMENT
═══════════════════════════════════════

The `ResearchState` dictionary acts as the shared memory for all agents. 

*   **`query`** (str): The user's main topic.
    *   *Producer:* `main.py`
    *   *Consumer:* Planner, Retriever, Critic, Writer
*   **`sub_questions`** (List[str]): Deconstructed angles to search.
    *   *Producer:* Planner
    *   *Consumer:* Search (Cycle 1 only)
*   **`search_results`** (List[Dict]): The raw, noisy scraped snippets. Uses `operator.add` to accumulate across retries.
    *   *Producer:* Search
    *   *Consumer:* Retriever
*   **`retrieved_documents`** (List[Dict]): The mathematically filtered, Top-K snippets matching the query. Replaced every cycle.
    *   *Producer:* Retriever
    *   *Consumer:* Critic, Writer
*   **`report`** (str): The final synthesized markdown.
    *   *Producer:* Writer
    *   *Consumer:* `main.py`
*   **`quality_score`** (float): The Critic's 0.0 to 1.0 assessment of `retrieved_documents`.
    *   *Producer:* Critic
    *   *Consumer:* `main.py` (observability)
*   **`critic_feedback`** (str): The Critic's textual reasoning.
    *   *Producer:* Critic
    *   *Consumer:* Search (for retry keyword extraction)
*   **`is_valid`** (bool): Deterministic flag set by Python (`score >= THRESHOLD`).
    *   *Producer:* Critic
    *   *Consumer:* Router (`route_after_critic`)
*   **`iteration_count`** (int): Tracks loop depth to prevent infinite cycles.
    *   *Producer:* Search (increments by 1)
    *   *Consumer:* Router, Search
*   **`sources`** (List[str]): Accumulated list of raw URLs scraped. Uses `operator.add`.
    *   *Producer:* Search
    *   *Consumer:* `main.py`
*   **`status`** (str): General phase indicator (e.g., `search_complete`, `validation_failed`).
    *   *Producer:* All Nodes
    *   *Consumer:* `main.py`

---

## SECTION 9 — RUNTIME VALIDATION RESULTS
═══════════════════════════════════════

During our Phase 3A runtime validation, the system successfully executed a multi-cycle, self-correcting research loop on the query: *"Future of Artificial Intelligence"*.

### Cycle 1
*   The Search node found 15 initial documents.
*   The Retriever indexed them and pulled the Top 5.
*   **Critic Evaluation:** It rejected the documents with a score of `0.6`, noting that while the documents provided a good overview, they lacked depth on societal implications and potential risks.
*   **Router Decision:** `RETRY` (Iteration 1 < MAX_ITERATIONS 3).

### Cycle 2
*   **Retry Search:** The Search agent read the feedback, extracted keywords like "societal implications" and "risks", and searched for those specific topics. 
*   It scraped 9 new documents, appending them to the Qdrant database (total 24).
*   **Critic Evaluation:** It rejected the newly retrieved Top 5 with a score of `0.6`, stating the new documents focused too heavily on education and lacked economic implications.
*   **Router Decision:** `RETRY` (Iteration 2 < 3).

### Cycle 3
*   **Retry Search:** The Search agent generated new queries based on the "economic implications" feedback, finding 9 more documents (total 33).
*   **Critic Evaluation:** The score remained at `0.6`. The Critic determined the sources were too promotional.
*   **Router Decision:** `REJECTED | MAX_ITERATIONS (3) reached`. 
*   **Graceful Degradation:** The safety valve triggered. Instead of looping infinitely, the Router forced the graph to the Writer node to synthesize the best report possible from the 33 indexed documents.

### Final Execution Metrics
*   **Final Status:** `writing_complete`
*   **Search Loops:** `3`
*   **Sources Found:** `32` (Deduplicated)
*   **Docs Retrieved:** `5`
*   **Final Quality Score:** `0.6`

**Conclusion:** The autonomous loop functioned flawlessly. The system actively sought out missing data, expanded its vector database on the fly, and successfully prevented an infinite loop.

---

## SECTION 10 — BUGS DISCOVERED
═══════════════════════════════════════

### 1. Critic Score Scale Mismatch
*   **Expected:** A float between 0.0 and 1.0.
*   **Received:** `6.5`
*   **Root Cause:** The LLM prompt requested a score, but the LLM assumed a 0 to 10 scale instead of 0 to 1, causing the Python deterministic logic (`score >= 0.70`) to improperly pass a failing score (`6.5 >= 0.70` evaluated to True).
*   **Fix:** Added strict Pydantic `Field(ge=0.0, le=1.0)` constraints to the schema, forcing an immediate validation error if the scale is violated. Additionally, reinforced the prompt with caps: `"YOU MUST OUTPUT A SCORE BETWEEN 0.0 AND 1.0"`.
*   **Lessons Learned:** Never trust an LLM to infer a scale without hard schema-level enforcement.

### 2. State Merge Issues in Streaming
*   **Symptoms:** When streaming the graph, terminal metrics for `search_results` were not accumulating; they were overwriting.
*   **Root Cause:** `graph.stream()` yields *partial* state updates (only the keys modified by the node). Using python's native `dict.update()` blindly overwrote the lists instead of appending to them.
*   **Fix:** Implemented a custom `merge_state()` helper function in `main.py` that mimics LangGraph's internal `operator.add` reducer, ensuring local tracking accurately matched the graph's internal history.

### 3. In-Memory Qdrant Volatility
*   **Symptoms:** `Collection 'deeptrace_research' not found` during the storage phase.
*   **Root Cause:** Using `QdrantClient(":memory:")` across multiple independent function calls inside the pipeline instantiated a brand new, blank database on every call, erasing the collection created seconds earlier.
*   **Fix:** Switched to persistent local disk storage (`path="./local_qdrant_storage"`) and implemented a Python Singleton in `qdrant_store.py` to prevent concurrent folder locking issues.

---

## SECTION 11 — DESIGN DECISIONS
═══════════════════════════════════════

1.  **Deterministic Thresholds (`QUALITY_THRESHOLD`)**
    *   *Decision:* Decoupled the binary `is_valid` decision from the LLM, moving it to Python (`score >= 0.70`).
    *   *Why:* Ensures strict predictability. If the system is too lenient or too harsh in production, we can adjust a single integer variable rather than playing a guessing game with prompt engineering.

2.  **Fail-Closed Validation**
    *   *Decision:* If the Critic LLM API call fails, `is_valid` defaults to `False` and `quality_score` to `0.0`.
    *   *Why:* "Fail-open" is dangerous. If the API is down, failing open would pass completely un-evaluated (and potentially harmful/hallucinated) context to the Writer. Failing closed forces a safe retry or triggers the `MAX_ITERATIONS` graceful shutdown.

3.  **Pydantic Constraints (`ge=0.0`, `le=1.0`)**
    *   *Decision:* Using Pydantic Field constraints rather than just prompt instructions.
    *   *Why:* Provides a strict runtime barrier against LLM hallucination or scale misinterpretation (like the `6.5` bug). If the LLM violates the rule, Pydantic throws an error, which triggers our Fail-Closed logic.

4.  **No Extra LLM Retry Generation**
    *   *Decision:* The Search Agent generates retry queries using regex and stop-word filtering in pure Python, rather than asking the LLM to rewrite the query.
    *   *Why:* It adheres to single-responsibility (Search shouldn't reason), reduces API latency, saves token costs, and ensures generated queries are highly indexed keywords rather than natural language sentences.

5.  **MAX_ITERATIONS Safety Valve**
    *   *Decision:* A hardcoded limit (`MAX = 3`) on the conditional router.
    *   *Why:* Autonomous loops run the risk of becoming infinite if the LLM cannot find the required data. This prevents API billing exhaustion and ensures the system always eventually returns a response to the user.

---

## SECTION 12 — INTERVIEW PREPARATION
═══════════════════════════════════════

**Q1. What is LangGraph, and how does it differ from standard LangChain?**
A: LangChain is for building linear chains (`Prompt | LLM | Output`). LangGraph is for building complex, cyclical state machines (DAGs) using nodes and edges. It manages a persistent global state across multiple agent interactions, enabling advanced loops and conditional routing.

**Q2. How do you prevent infinite loops in a multi-agent system?**
A: By introducing a state variable like `iteration_count`. The node responsible for looping (e.g., Search) increments the count on every execution. The conditional router inspects this count and forces an exit path (like moving to the Writer node) when a `MAX_ITERATIONS` threshold is hit.

**Q3. What is the difference between RAG and a standard LLM query?**
A: A standard query relies on the LLM's internal, static training weights, risking hallucinations. RAG (Retrieval-Augmented Generation) first searches an external database for facts, injects them into the prompt, and forces the LLM to synthesize an answer based *only* on that injected context.

**Q4. Why did you choose Qdrant?**
A: It is natively built in Rust, ensuring extreme performance for vector similarity calculations via HNSW graphs. Crucially for development, it supports a fully local, disk-based mode that doesn't require Docker or external cloud accounts.

**Q5. What is the purpose of a Critic Agent?**
A: It acts as an autonomous QA gatekeeper. Generative AI is biased toward creation. The Critic separates the creation task from the evaluation task, preventing the system from synthesizing reports based on irrelevant or hallucinated context.

**Q6. What is a Conditional Edge in LangGraph?**
A: Unlike a static edge (`add_edge("A", "B")`), a conditional edge executes a Python router function. Based on the global state (e.g., `is_valid == True`), the router dynamically dictates which node should execute next.

**Q7. Explain the "Lost in the Middle" phenomenon.**
A: It is an inherent flaw in large context windows where an LLM accurately recalls data at the very beginning and very end of a prompt, but ignores data buried in the middle. We mitigate this using a Vector DB to retrieve only the Top-5 most relevant chunks.

**Q8. Why use `with_structured_output` and Pydantic?**
A: To force non-deterministic LLMs to output predictable, parseable JSON. It allows us to seamlessly map LLM outputs to Python types (like floats for math operations) without writing fragile string parsers (Regex).

**Q9. What is "Fail-Closed" design?**
A: In security and systems engineering, if a process fails, it should fail into a safe state. If our Critic API call fails, we assume the data is *invalid* (closed) rather than assuming it's *valid* (open).

**Q10. How did you implement Search retries without adding more LLM calls?**
A: By using deterministic keyword extraction in Python. We stripped out "editorial filler" (like "Coverage is missing") and stop words from the Critic's feedback, extracted the core nouns, and appended them to the original query.

**Q11. What is an Embedding?**
A: It is a high-dimensional mathematical vector (array of floats) representing the underlying semantic meaning of a piece of text.

**Q12. What is Cosine Similarity?**
A: A mathematical metric that determines how close two vectors are in multidimensional space by measuring the cosine of the angle between them. The closer the vectors, the more semantically similar the original texts.

**Q13. What is the difference between `graph.invoke()` and `graph.stream()`?**
A: `invoke()` is synchronous and hides internal state transitions until the final node completes. `stream()` is a generator that yields partial state updates as every node finishes, enabling real-time observability.

**Q14. How does LangGraph handle state accumulation?**
A: By using reducers like `operator.add` in the `TypedDict`. When a node returns `{"search_results": [...] }`, LangGraph doesn't overwrite the existing list; it extends it.

**Q15. Why did you separate evaluation (LLM) from decision (Python)?**
A: For determinism. Asking an LLM to output a boolean `True/False` for validity is risky due to prompt variation. Asking it for a score and using Python (`score >= 0.70`) gives us absolute, easily tunable control.

**Q16. What is a Singleton pattern and why use it for local Qdrant?**
A: A Singleton ensures only one instance of a class exists. Local Qdrant locks the physical directory on disk. If multiple Python functions instantiate new clients, they crash fighting for the lock. The Singleton prevents this.

**Q17. How does DeepTrace handle deduplication of sources?**
A: The Search Agent wraps the accumulated URL sources in `list(set())` before returning the state update, guaranteeing uniqueness.

**Q18. Describe the complete DeepTrace loop.**
A: Planner -> Search -> Retriever -> Critic -> Router. If Rejected -> Search -> Retriever -> Critic. If Approved -> Writer -> End.

**Q19. What is "Graceful Degradation"?**
A: When a system encounters an error or hits a hard limit (like `MAX_ITERATIONS`), it doesn't crash. It falls back to the safest possible operation—in our case, forcing the Writer to synthesize whatever partial data it managed to find.

**Q20. Why avoid the `DuckDuckGoSearchResults` tool?**
A: LangChain's conversational tools concatenate outputs into a single, messy string. By using the `DuckDuckGoSearchAPIWrapper` directly, we get a native Python `List[Dict]`, which guarantees clean structural integrity for our Vector DB metadata.

---

## SECTION 13 — CURRENT PROJECT STATUS
═══════════════════════════════════════

*   **Phase 1 (Core Pipeline):** COMPLETE
*   **Phase 2 (Persistent Memory & RAG):** COMPLETE
*   **Phase 3A (Critic & Conditional Routing):** COMPLETE

**Current Capabilities:**
DeepTrace is now a fully functional, self-correcting RAG system. It can autonomously break down broad questions, scrape the web, build its own vector database on the fly, evaluate the semantic quality of its findings, and loop backward to discover missing facts before synthesizing a final, cited markdown report.

---

## SECTION 14 — FUTURE ROADMAP
═══════════════════════════════════════

### Phase 3B: Advanced Retry Optimization
*   Implement an LLM-powered **Query Optimizer Agent** to replace our deterministic Python keyword extractor, enabling highly nuanced, context-aware retry searches.

### Phase 4: Source Ranking & Memory Expansion
*   Implement **Domain Authority Scoring** (e.g., heavily weighting `.gov` or `.edu` domains over random blogs).
*   Add persistent cross-session memory so DeepTrace remembers research from previous user sessions.

### Phase 5: Production Deployment
*   Migrate from local Qdrant disk storage to a centralized Qdrant Cloud or Docker Server deployment.
*   Implement a full React/Streamlit Web UI.
*   Introduce "Human-in-the-Loop" conditional edges (pausing the graph to ask the user if they want to approve the Critic's feedback before retrying).