# ResearchGPT (DeepTrace) - Complete Technical Audit Report

TOTAL FILES DISCOVERED: 134
TOTAL RELEVANT FILES INSPECTED: 29
FILES NOT INSPECTED: `__pycache__` directories, `venv` environment files, compiled python files (`.pyc`), and local qdrant storage binaries.
REASON FOR ANY FILE NOT INSPECTED: These are environment, generated, or binary files that do not contain source code or configuration authored by the developer.

INSPECTED FILE PATHS:
- `README.md`
- `main.py`
- `app/agents/planner.py`
- `app/agents/critic.py`
- `app/agents/search.py`
- `app/agents/writer.py`
- `app/agents/query_optimizer.py`
- `app/agents/retriever.py`
- `app/graph/workflow.py`
- `app/graph/state.py`
- `app/api/main.py`
- `app/core/llm.py`
- `app/database/connection.py`
- `app/database/models.py`
- `app/database/postgres.py`
- `app/search/providers.py`
- `app/vectorstore/qdrant_store.py`
- `app/evaluation/langsmith_eval.py`
- `app/evaluation/ragas_eval.py`
- `app/frontend/app.py`
- `docker/docker-compose.yml`
- `docker/Dockerfile`
- `requirements.txt`
- `.env.example`
- `alembic.ini`
- `alembic/env.py`
- `alembic/versions/706f944a464f_create_research_runs_table.py`
- `tests/test_agents.py`
- `tools/validate_external_services.py`

---

## 1. Executive Summary
This audit validates the technical implementation of **ResearchGPT** (formerly DeepTrace). The codebase presents a moderately sophisticated, fully functional **multi-agent LangGraph workflow** that executes autonomous web research, RAG (Retrieval-Augmented Generation), and critical self-evaluation. 

While the core AI pipeline is **Fully Implemented** and demonstrates advanced patterns like LLM fallback chains, dynamic search intent classification, and Trust-Tier reranking, the surrounding infrastructure (PostgreSQL database, FastAPI backend, Streamlit frontend) is completely **Placeholder / Unused**.

## 2. What ResearchGPT Is
ResearchGPT is a CLI-based autonomous AI research assistant. Given a topic, it orchestrates 6 distinct AI agents to break down the query, scrape the web, build a local vector database, evaluate the quality of its own research, self-correct if needed, and synthesize a hallucination-free markdown report with citations.

## 3. Real-World Problem
Normal web searching (e.g., Google or ChatGPT Web Search) is linear and shallow, often suffering from hallucinations or missing deep contextual facts. ResearchGPT solves this by employing an automated multi-agent system that mimics a human researcher: it breaks a broad question into sub-questions, executes parallel searches, cross-references facts mathematically via embeddings, grades its own findings, and iterates if the research is insufficient. 

## 4. Why This Architecture Exists
The LangGraph DAG (Directed Acyclic Graph) architecture exists to provide **stateful, cyclic routing**. Unlike standard linear LangChain chains, LangGraph allows the system to loop back to the search phase if the `Critic` agent determines the gathered information is poor. Qdrant is used to solve the "Lost in the Middle" LLM problem by converting noisy web scrapes into top-K semantic chunks, ensuring the final writer agent only sees highly relevant facts.

## 5. Complete Folder Structure
```text
/app
  /agents
    planner.py
    search.py
    retriever.py
    critic.py
    query_optimizer.py
    writer.py
  /api
    main.py (Empty)
  /core
    llm.py
  /database
    connection.py
    models.py
    postgres.py (Empty)
  /evaluation
    langsmith_eval.py (Empty)
    ragas_eval.py (Empty)
  /frontend
    app.py (Empty)
  /graph
    state.py
    workflow.py
  /search
    providers.py
  /vectorstore
    qdrant_store.py
/alembic (DB migrations)
/docker (Compose & Dockerfile)
main.py (CLI Entrypoint)
```

## 6. File-by-File Explanation

**FILE:** `main.py`
**PURPOSE:** The CLI entry point that initializes the Qdrant collection, builds the LangGraph, executes the graph with an initial state, and streams logs to the console.
**WHAT IT DOES:** Generates a `run_id`, runs the workflow, manually merges state updates for console printing, and outputs the final report.

**FILE:** `app/graph/workflow.py`
**PURPOSE:** LangGraph orchestration.
**WHAT IT DOES:** Wires the nodes together. Uses a conditional edge (`route_after_critic`) to decide whether to route to the Writer (if valid) or Query Optimizer (if rejected and iterations < 3).

**FILE:** `app/graph/state.py`
**PURPOSE:** Defines the shared memory schema.
**WHAT IT DOES:** Uses `TypedDict` with `Annotated[..., operator.add]` to define how state updates are merged (e.g., appending to lists vs replacing strings).

**FILE:** `app/core/llm.py`
**PURPOSE:** Centralized LLM configuration.
**WHAT IT DOES:** Instantiates OpenRouter models using `ChatOpenAI` and implements a 4-model fallback chain (`with_fallbacks`).

**FILE:** `app/search/providers.py`
**PURPOSE:** Search provider abstraction.
**WHAT IT DOES:** Implements `TavilySearchProvider`, `DuckDuckGoSearchProvider`, and `ChainedSearchProvider`. It includes a keyword-based query classifier to select optimal `include_domains` for Tavily.

**FILE:** `app/vectorstore/qdrant_store.py`
**PURPOSE:** Local vector database integration.
**WHAT IT DOES:** Embeds text using FastEmbed (`BAAI/bge-small-en-v1.5`), stores payloads in local Qdrant, and retrieves them using `search_similar` filtered by `run_id`.

**FILE:** `app/database/models.py` & `connection.py`
**PURPOSE:** SQLAlchemy ORM models.
**WHAT IT DOES:** Defines a `ResearchRun` table to store queries, state, and reports. *(Configured but unused in main flow)*.

## 7. System Architecture
**Simple Explanation:** 
The system is like a newsroom. The Planner is the editor assigning 5 specific tasks. The Searcher goes out and gets raw articles. The Retriever reads the articles, highlights the best quotes, and files them in a cabinet (Qdrant). The Critic acts as the QA editor, reviewing the quotes. If they are bad, they send the Searcher back out with new instructions (Query Optimizer). If good, the Writer writes the final story.

**Technical Explanation:**
- **Agent/LangGraph Layer:** A DAG where nodes are Python functions and edges define data flow.
- **Search Layer:** `ChainedSearchProvider` orchestrates an API call to Tavily. If it fails, it falls back to a DuckDuckGo web scraper. 
- **Retrieval Layer:** The `retriever_node` bridges search and synthesis. It takes raw search text, generates local embeddings via FastEmbed, stores them in local Qdrant, immediately similarity-searches the same Qdrant collection, applies a Python-based domain authority multiplier, and extracts the top 5 chunks.
- **LLM Layer:** Centralized `get_llm()` uses `langchain_openai.ChatOpenAI` configured for OpenRouter, utilizing native `.with_fallbacks()` for high availability.

## 8. LangGraph Architecture
- **State:** `ResearchState`
- **Entry:** `START -> planner`
- **Nodes:** `planner`, `search`, `retriever`, `critic`, `query_optimizer`, `writer`
- **Edges:** Linear until `critic`. After `critic`, `add_conditional_edges` uses `route_after_critic()`.
- **Iteration Logic:** If `is_valid` is False and `iteration_count < MAX_ITERATIONS (3)`, routes to `query_optimizer` -> `search`. 

## 9. ResearchState Deep Dive
- `run_id` (str): UUID for vector isolation.
- `query` (str): Original user query.
- `sub_questions` (List[str]): 5 questions from Planner.
- `search_results` (Annotated[List, operator.add]): Raw scrapes. Appends across cycles.
- `retry_queries` (List[str]): 3 questions from Optimizer.
- `retrieved_documents` (List[dict]): Curated top 5 facts. Overwritten each cycle.
- `sources` (Annotated[List, operator.add]): URLs.
- `iteration_count` (int): Replaces previous value.
- `critic_feedback` (str): Explains rejection.
- `quality_score` (float): Numerical evaluation.
- `is_valid` (bool): Boolean trigger for routing.
- `report` (str): Final markdown output.
- `status` (str): UI/Tracking indicator.

## 10. Planner Agent
- **Problem:** Broad queries yield shallow searches.
- **Input:** User query.
- **Prompt:** Instructs the LLM to output exactly 5 angles.
- **Validation:** Uses `PydanticOutputParser` to enforce JSON schema. Python raises `ValueError` if `len != 5`.

## 11. Search Agent
- **Logic:** Reads `sub_questions` (or `retry_queries` if cycle > 0).
- **Execution:** Calls `provider.search(q)`. The provider is `ChainedSearchProvider` (Tavily -> DDG).
- **Output:** Normalizes results to `{question, title, snippet, source}` and appends to `search_results`.

## 12. RAG Pipeline
Documents -> Local FastEmbed CPU generation (`BAAI/bge-small-en-v1.5`, 384 dims) -> Qdrant Local SQLite Storage -> Cosine Similarity Search -> Top 20 retrieved -> Trust-Tier Domain Multiplier applied -> Top 5 extracted -> Writer Agent.

## 13. Qdrant
- **Persistent/In-Memory:** PERSISTENT. It uses `path="./local_qdrant_storage"`, which writes to SQLite on disk.
- **Isolation:** `run_id` is stored in the payload. `search_similar` filters by this `run_id` to prevent cross-contamination between script executions.

## 14. Critic Agent
- **Input:** Formatted string of `retrieved_documents`.
- **Prompt:** Instructs LLM to grade 0.0 to 1.0. 
- **Format:** Uses `.with_structured_output(CriticEvaluation)` for strict Pydantic parsing.
- **Logic:** Python code sets `is_valid = quality_score >= 0.70`.

## 15. Feedback/Iteration Loop
If `is_valid` is False:
1. `route_after_critic` directs to `query_optimizer`.
2. Optimizer generates 3 new targeted queries based on `critic_feedback`.
3. Graph routes back to `search`.
4. Increments `iteration_count`. Cap is 3.

## 16. Writer Agent
- **Input:** Minimal string of ONLY `retrieved_documents` (Top 5).
- **Prompt:** Strict markdown formatting instructions and anti-hallucination rules.
- **Output:** Synthesized text assigned to `report`.

## 17. LLM/OpenRouter
- **Abstraction:** `app.core.llm.get_llm()`
- **Strategy:** Configured for `meta-llama/llama-3.1-8b-instruct`. 
- **Fallbacks:** Uses `.with_fallbacks()` pointing to Mistral, Gemma, and Phi. If the Llama API fails, LangChain instantly attempts Mistral.

## 18. PostgreSQL
- **Implementation:** `app/database/models.py` defines tables. Alembic handles migrations. 
- **Usage:** **NOT ACTUALLY USED.** `main.py` never instantiates the DB session, and no agent writes to it. 

## 19. FastAPI
**Implementation Status:** PLACEHOLDER. `app/api/main.py` is empty.

## 20. Streamlit
**Implementation Status:** PLACEHOLDER. `app/frontend/app.py` is empty.

## 21. End-to-End Data Flow
User runs CLI -> `main.py` initializes State -> Planner -> Search (Tavily/DDG) -> Retriever (Qdrant Embed & Query) -> Critic -> (Conditional Loop if < 0.7) -> Writer -> `main.py` prints to terminal.

## 22. Complete Execution Example
1. User runs `python main.py`. State initialized with "Future of Artificial Intelligence".
2. Planner generates 5 sub-questions.
3. Search agent executes 5 searches via Tavily, returning 15 raw snippets.
4. Retriever embeds 15 snippets into Qdrant, searches Qdrant for the original query, pulls top 20, reranks based on domains (e.g. `nature.com` gets 1.5x boost), and saves top 5 to state.
5. Critic grades the 5 snippets. Gives it a 0.8.
6. Graph decides `0.8 > 0.70`, routes to Writer.
7. Writer produces a markdown report.
8. `main.py` logs the report.

## 23. Technology Stack
- **Python 3.13** (Actual)
- **LangGraph & LangChain** (Actual)
- **OpenRouter** (Actual)
- **Qdrant & FastEmbed** (Actual - Local)
- **Tavily & DuckDuckGo** (Actual)
- **PostgreSQL & Alembic** (Configured, Unused)
- **FastAPI, Streamlit, LangSmith, Ragas** (Empty placeholders)

## 24. Feature Inventory
| Feature | Implementation Status | Files | How it works |
|---------|------------------------|-------|--------------|
| Agent Orchestration | Fully Implemented | `workflow.py` | StateGraph DAG with conditional routing |
| Vector Retrieval | Fully Implemented | `retriever.py`, `qdrant_store.py` | FastEmbed to local Qdrant |
| Trust-Tier Reranking | Fully Implemented | `retriever.py` | Hardcoded domain multipliers |
| LLM Fallbacks | Fully Implemented | `llm.py` | LangChain Runnable fallbacks |
| Database Persistence | Configured / Unused | `models.py` | Alembic exists, but no writes occur |
| API & UI | Placeholder | `api/main.py`, `frontend/app.py` | Empty files |

## 25. Actual Technical Contributions
**WHAT:** Multi-Agent LangGraph Architecture
**HOW:** Implemented a cyclical DAG orchestration using `StateGraph`, enabling agents to route dynamically based on autonomous QA checks.
**TECHNOLOGY:** Python, LangGraph

**WHAT:** Fault-Tolerant LLM Gateway
**HOW:** Centralized LLM configuration using OpenRouter with `.with_fallbacks()` across 4 distinct models to ensure high availability.
**TECHNOLOGY:** LangChain, OpenRouter

**WHAT:** RAG Pipeline with Trust-Tier Reranking
**HOW:** Built a local embedding pipeline using FastEmbed and Qdrant. Implemented a post-retrieval reranker that multiplies semantic scores by domain authority (e.g., .gov, .edu) to prioritize factual accuracy.
**TECHNOLOGY:** Qdrant, FastEmbed

**WHAT:** Search Intent Classification
**HOW:** Created a deterministic query classifier that injects dynamic domain whitelists into Tavily API calls based on whether a query is scientific, news, or policy.
**TECHNOLOGY:** Tavily API

## 26. Resume-Relevant Raw Facts
- Designed a **6-agent** autonomous research pipeline (Planner, Search, Retriever, Critic, Optimizer, Writer).
- Implemented an LLM fallback chain utilizing **4 separate models** (Llama, Mistral, Gemma, Phi) to ensure zero downtime.
- Configured local RAG using **384-dimensional** embeddings (BGE-Small) stored in local Qdrant.
- Engineered a deterministic Trust-Tier reranking algorithm that filters a pool of **20 fetched documents** down to the **top 5 most authoritative sources**.
- Enforced strict JSON validation using Pydantic, mandating exactly **5 sub-questions** per plan and **3 optimized queries** per retry loop.
- Built a QA Critic loop that forces a retry if the semantic quality score drops below **0.70**, capped at **3 iterations** to prevent infinite loops.

## 27. Documentation vs Code
| Claim | Found in Doc | Found in Code | Status |
|-------|--------------|---------------|--------|
| Phase 1-3 (LangGraph, RAG, Critic) | Yes | Yes | Validated |
| Phase 4 (Trust-Tier) | Yes (Mentioned as coming) | Yes | Validated (Actually Implemented) |
| Phase 5-8 (Postgres, FastAPI, UI) | Yes (Mentioned as coming) | No | Placeholders / Unused |

## 28. Code Quality Issues
- **Dead Code:** The PostgreSQL models, connection logic, and Alembic migrations exist but are never invoked in `main.py` or the graph nodes.
- **Empty Files:** `app/api/main.py`, `app/frontend/app.py`, `app/database/postgres.py`, and evaluation scripts are entirely empty.
- **Redundant State Merging:** `main.py` implements a manual `merge_state` function to replicate LangGraph's reducer logic just for printing console logs.
- **Qdrant Initialization:** In `qdrant_store.py`, `store_documents` checks and creates the collection inside the workflow execution. While functional, this should ideally be handled at startup to avoid runtime latency.
- **Error Swallowing:** The `search_similar` function returns an empty list `[]` on failure. While this acts as graceful degradation, it swallows the error entirely without appending it to the `errors` state array.

## 29. Technical Limitations & Improvements
- **Missing Persistence:** The pipeline's outputs disappear once the CLI terminates. Needs to integrate the existing SQLAlchemy models.
- **Synchronous Execution:** The agents and search API calls run sequentially. Using `asyncio` and `AsyncStateGraph` would drastically speed up the parallel 5 sub-question searches.

## 30. Interview Preparation (Top 20 Things to Understand)
1. **LangGraph vs LangChain:** Understand that LangGraph allows cycles and stateful memory, whereas LangChain is traditionally linear.
2. **State Reducers:** Be able to explain why `Annotated[List, operator.add]` is used so that lists are appended to rather than overwritten.
3. **Pydantic Output Parsers:** Explain how you forced the LLM to return valid JSON (e.g., exactly 5 questions).
4. **Vector Databases (Qdrant):** Know that Qdrant stores arrays of floats (embeddings) and uses Cosine Similarity to find distances.
5. **Embedding Models (FastEmbed):** Explain that `bge-small` runs locally on the CPU, converting text to 384-dimension vectors.
6. **The "Lost in the Middle" Problem:** Explain why RAG is used (to filter out noise and only feed the top 5 chunks to the Writer).
7. **Trust-Tier Reranking:** Describe your custom logic of multiplying the Qdrant score by 1.5x for `.gov`/`.edu` domains.
8. **LLM Fallback Chains:** Detail how `.with_fallbacks()` routes around OpenRouter rate limits.
9. **Conditional Edges:** Explain how the Critic returns a boolean `is_valid` which triggers the router function to loop or finish.
10. **Query Intent Classification:** Explain the keyword-based approach in `providers.py` to route search parameters.
11. **UUID Run Isolation:** Explain how `run_id` is attached to Qdrant payloads to prevent queries from bleeding across different application runs.
12. **Graceful Degradation:** Explain how the system continues even if Tavily goes down (falling back to DuckDuckGo). 
13. **Hallucination Prevention:** Be able to articulate how the Writer agent is strictly instructed to use ONLY the `retrieved_documents`.
14. **Deterministic Routing vs LLM Routing:** Explain why the LLM only outputs a score (0.0-1.0), but the Python code (`>= 0.70`) actually makes the routing decision.
15. **SQLAlchemy vs Alembic:** (Even though unused, you configured it). SQLAlchemy defines the models; Alembic tracks the schema migrations.
16. **Why SQLite for Qdrant?** Explain that `path="./local_qdrant_storage"` bypasses the need for a Dockerized Qdrant instance for easier local deployment.
17. **Deduplication Logic:** Explain how the retriever prevents two chunks from the exact same URL from occupying the top 5 spots.
18. **Prompt Engineering:** Know how you structured prompts with clear roles, rules, and injected context.
19. **Fail-Closed Logic:** Know that if the Critic API fails, you route to an error state rather than blindly approving unverified data.
20. **Synchronous vs Asynchronous:** Be prepared to explain that the current implementation is synchronous, and how you would migrate it to Async (using `aiohttp` and async LangGraph nodes).
