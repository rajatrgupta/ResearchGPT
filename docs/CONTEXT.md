# DeepTrace Context and Memory

## Project Overview
DeepTrace is a production-grade, LangGraph-powered Multi-Agent Research System. When provided with a research query, it autonomously orchestrates a workflow of specialized agents to break down the query, search the web for up-to-date information, validate findings, and synthesize a structured, well-cited markdown report. It leverages Retrieval-Augmented Generation (RAG) to ensure responses are grounded in factual, external data rather than relying solely on the LLM's internal knowledge.

## Complete Tech Stack
- **Language**: Python 3.11
- **Agent Framework**: LangGraph + LangChain
- **LLMs**: OpenRouter API (Fallback strategy: Llama 3.1 8B -> Mistral 7B -> Gemma 2 9B -> Phi-3 Mini)
- **Web Search**: DuckDuckGo Search API (`ddgs`)
- **Vector Database**: Qdrant (Planned for Phase 2)
- **Relational DB**: PostgreSQL (Planned)
- **Backend API**: FastAPI (Planned)
- **Frontend**: Streamlit (Planned)
- **Evaluation & Tracing**: RAGAS + LangSmith (Planned)
- **Data Validation**: Pydantic

## Folder Structure
```text
DeepTrace/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py     (Breaks down query into 5 sub-questions)
│   │   ├── search.py      (Fetches structured web results)
│   │   ├── critic.py      (Empty placeholder)
│   │   └── writer.py      (Synthesizes findings into markdown report)
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm.py         (Centralized LLM configuration with fallbacks)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py       (Defines the ResearchState TypedDict)
│   │   └── workflow.py    (Compiles the LangGraph DAG)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres.py
│   │   └── models.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── qdrant_store.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_eval.py
│   │   └── langsmith_eval.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── frontend/
│       └── app.py
├── tests/
│   └── test_agents.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── CONTEXT.md             (This file)
└── main.py                (Entry point for execution)
```

## Phase 1 Accomplishments
Phase 1 focused on building the core LangGraph multi-agent pipeline (Planner -> Search -> Writer).

- **`app/graph/state.py`**: Defines the `ResearchState` TypedDict. Key Concept: State management, using `operator.add` for accumulating lists to prevent data overwrite across iterations. Added a custom reducer `merge_unique_sources` for deduplicating URLs globally.
- **`app/core/llm.py`**: Configures the ChatOpenAI client via OpenRouter. Key Concept: Fault-tolerant LLM gateway using LangChain's `.with_fallbacks()` across multiple free models to seamlessly handle rate limits and endpoint outages.
- **`app/agents/planner.py`**: The Planner Agent. Key Concept: Prompt engineering to split overarching queries into exactly 5 focused sub-questions. Enforced by `PydanticOutputParser` to guarantee structured JSON output.
- **`app/agents/search.py`**: The Search Agent. Key Concept: Using `DuckDuckGoSearchAPIWrapper` to fetch *structured* search results (title, snippet, link) instead of raw text strings. Implements robust error handling so a single failed sub-search doesn't crash the pipeline.
- **`app/agents/writer.py`**: The Writer Agent. Key Concept: Synthesis over summarization. Uses the RAG pattern to write a structured, comprehensive markdown report with inline citations based exclusively on the retrieved context snippets.
- **`app/graph/workflow.py`**: The LangGraph Orchestrator. Key Concept: Defining Nodes (agents) and Edges (transitions) to compile a Directed Acyclic Graph (DAG) representing the stateful execution flow.
- **`main.py`**: The entry point script. Key Concept: Loading environment variables with `dotenv`, initializing the `ResearchState`, invoking the LangGraph pipeline, and printing the final status and synthesized report.

## Current Working State
When `python main.py` is executed:
1. `dotenv` loads the `OPENROUTER_API_KEY` into the environment.
2. The LangGraph workflow is compiled.
3. The initial state is defined with the query "Future of Artificial Intelligence".
4. The Planner Node queries the OpenRouter API to generate 5 specific sub-questions.
5. The Search Node iterates through those 5 questions, querying DuckDuckGo, and appending structured dictionaries (title, snippet, link) into the global state.
6. The Writer Node formats the gathered snippets into a context block and prompts the LLM to synthesize a final markdown report containing inline citations.
7. The terminal outputs the final operational status, the number of search loops, the number of sources scraped, and the final structured markdown report.

## Important Architectural Decisions
- **Centralized LLM Configuration (`app/core/llm.py`)**: To ensure all agents share the exact same API keys, temperature settings, and model fallback logic, preventing code duplication and configuration drift.
- **Model Fallbacks**: Free OpenRouter models are prone to rate-limiting and timeouts. Relying on a single model creates a single point of failure. The `with_fallbacks()` strategy guarantees high availability.
- **Strict Pydantic Parsing in Planner**: LLMs frequently hallucinate formatting. By enforcing strict JSON output and raising a hard error if exactly 5 questions aren't returned, we ensure downstream nodes do not crash or process malformed data.
- **Structured Search Output over Raw Strings**: Instead of returning a giant concatenated string of scraped text, we extract specific dictionaries (title, snippet, link) from the Search API. This allows the Writer to accurately cite sources and explicitly prepares the data for rich metadata filtering in Qdrant.
- **Using API Wrapper Instead of Search Tool**: The LangChain `DuckDuckGoSearchResults` Tool wrapper forces a string output which requires brittle string parsing to recover lists. Using the lower-level `DuckDuckGoSearchAPIWrapper` returns a native Python list of dictionaries, ensuring 100% reliable structured output.

## Phase 2 Roadmap (Next Steps)
Phase 2 will integrate a Vector Database (Qdrant) to give our agents persistent semantic memory.
- We will set up a local Qdrant instance (likely via Docker).
- The Search Agent will be updated to generate vector embeddings of the retrieved snippets and insert them into Qdrant alongside metadata (title, link).
- The Writer (and future Critic) will be updated to query Qdrant to semantically retrieve the most relevant context rather than holding all scraped text inside the LangGraph memory state dictionary.

## Errors Faced and Solutions
1. **Langchain duckduckgo-search Import Error**:
   - **Error**: `Could not import ddgs python package...`
   - **Cause**: The underlying PyPI package for DuckDuckGo search was renamed/restructured to `ddgs`, breaking older LangChain community wrappers.
   - **Solution**: Explicitly installed the modern `ddgs` package (`pip install ddgs`) to resolve the underlying dependency mismatch.
2. **Missing Environment Variable Loader**:
   - **Error**: `OPENROUTER_API_KEY environment variable is not set.`
   - **Cause**: `os.environ.get()` was called in `core/llm.py` before the `.env` file was actually loaded into the runtime.
   - **Solution**: Added `from dotenv import load_dotenv; load_dotenv()` at the very top of `main.py`, guaranteeing it executes before any LangChain modules initialize.
3. **Unsafe String Parsing with `eval()`**:
   - **Error**: An early iteration of the Search agent used `eval()` to parse stringified search tool output back into a Python list.
   - **Cause**: Standard LangChain tools often serialize output to strings for conversational LLMs to read.
   - **Solution**: Refactored to use the lower-level API Wrapper (`DuckDuckGoSearchAPIWrapper`), bypassing string serialization entirely and closing a critical Remote Code Execution vulnerability.