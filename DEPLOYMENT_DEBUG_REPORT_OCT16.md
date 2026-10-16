# ResearchGPT Deployment Debug & Work Status Report
**Date:** October 16, 2026  
**Repository:** `rajatrgupta/ResearchGPT`  
**Live Frontend:** https://researchgpt-ai.streamlit.app/  
**Live Backend:** https://researchgpt-backend-ygh8.onrender.com  

---

## 1. Executive Summary

Today we performed a deep forensic investigation into why **ResearchGPT** stopped generating research reports after deployment. We identified and fixed the root causes that broke the local and deployed execution pipelines, verified complete report generation locally, and pushed the fixes to GitHub.

---

## 2. Root Cause Analysis (What Went Wrong & Why)

### Issue 1: Vector Store Embeddings 401 Unauthorized (The Primary Blocker)
- **What happened:** In previous deployment commits, local `FastEmbed` embeddings were replaced with `langchain_huggingface.HuggingFaceEndpointEmbeddings` using a placeholder token (`hf_your_fallback_token_if_needed`).
- **Why it failed:** Hugging Face's serverless inference router (`router.huggingface.co`) mandates a valid authenticated `HF_TOKEN`. Requests failed immediately with `401 Unauthorized: Invalid username or password`.
- **Downstream effect:** Every attempt by the Retriever agent to store search results in Qdrant threw an exception. `retrieved_documents` returned empty (`[]`). The Critic scored this as `0.0` (*"No documents retrieved. Complete failure"*). After 3 retry loops, the Writer agent aborted synthesis with *"Insufficient Data / Exhausted Research Limits"*.

### Issue 2: OpenRouter Fallback 402 Error (Credit Reservation Limit)
- **What happened:** When Google Gemini rate limits (429 `RESOURCE_EXHAUSTED` / 503 `UNAVAILABLE`) were hit and the system attempted to fall back to OpenRouter (`meta-llama/llama-3.1-8b-instruct`), `ChatOpenAI.with_structured_output` was invoked without an explicit `max_tokens` cap.
- **Why it failed:** OpenRouter calculated the required credit reservation based on the model's full context window (117,920 tokens), which exceeded the available balance. It threw `402 - This request requires more credits, or fewer max_tokens. You requested up to 117920 tokens, but can only afford 60817`.
- **Downstream effect:** The fallback provider crashed, leaving the system with no working LLM.

### Issue 3: Deprecated/Non-Existent Gemini Models in Failover Chain
- **What happened:** The model list in `app/core/llm.py` contained deprecated models (`gemini-1.5-flash-001`, `gemini-1.5-flash-002`, `gemini-1.5-flash-latest`, `gemini-1.5-pro-latest`, `gemini-pro`, `gemini-2.5-flash`).
- **Why it failed:** In Google's v1beta API, these return immediate `404 NOT_FOUND` errors, resulting in wasted latency and retries across multiple keys before reaching real models.

### Issue 4: Planner Output Parser Echoing Schema
- **What happened:** The Planner node was using text-based `PydanticOutputParser` rather than native structured output function calling.
- **Why it failed:** Certain fallback models echoed the JSON schema definition instead of populating the `questions` array, throwing a Pydantic validation failure.

### Issue 5: Streamlit Polling Loop Fragility
- **What happened:** In `app/frontend/app.py`, any single transient network read timeout or rate-limit code (429) immediately broke out of the polling `while` loop (`break`), freezing the user interface.

### Issue 6: Why the Live Render Server Got Stuck on "search"
- **What happened:** In the Render server logs, the background task ran:
  1. `[Service] Run 8ad3be26... started processing.`
  2. `[Planner] COMPLETE`
  3. `[TavilySearchProvider] COMPLETE in 3.52s (3 results) ...`
  4. `[Service] Run 8ad3be26... completed node: search`
  5. `[Retriever] START Qdrant storage...`
  6. `[Embeddings] Initializing FastEmbed (BAAI/bge-small-en-v1.5)...`
  7. `Fetching 5 files: 100%|██████████| 5/5 [00:05<00:00, 1.07s/it]`
  8. `==> Running 'alembic upgrade head && uvicorn app.api.main:app --host 0.0.0.0 --port 10000'`
- **Why it was stuck:** Right as FastEmbed finished downloading the model and loaded the ONNX weights into RAM, Render **restarted the container** (either due to a new Git commit deployment rollout or an Out-Of-Memory / OOM event on Render's 512MB free tier). 
- Because the background thread was terminated by the process restart, the task was orphaned in PostgreSQL with `status="running"` and `stage="search"`. Every subsequent GET poll simply read that static database row.

---

## 3. Work Completed Today (Code Changes & Commits)

### Commits Pushed to `origin/main`

1. **Commit `962bfa3`**: `fix(retrieval,llm): resolve report generation failures by restoring resilient embeddings and fixing OpenRouter fallback quota limits`
   - Replaced broken `HuggingFaceEndpointEmbeddings` with `ResilientEmbeddingModel` in `app/vectorstore/qdrant_store.py`.
   - Added support for `GEMINI_API_KEY`, `GOOGLE_API_KEYS`, and `OPENROUTER_API_KEY`.
   - Set `max_tokens=4096` and `timeout=30.0` on OpenRouter `ChatOpenAI` instances.
   - Added `fastembed>=0.2.7` to `requirements.txt`.
   - Supported both `postgres://` and `postgresql://` in `app/database/connection.py`.

2. **Commit `392bfde`**: `fix(planner,core): enforce structured output for planning node and ensure automatic dotenv initialization`
   - Migrated Planner node to use `get_llm(structured_schema=SubQuestions)`.
   - Added deterministic high-quality fallback questions in case of unexpected LLM provider timeouts.
   - Added automatic `load_dotenv()` execution to `app/core/llm.py` and `app/vectorstore/qdrant_store.py`.

3. **Commit `8ad7315`**: `fix(frontend): improve polling loop resilience against network blips and rate limits`
   - Made Streamlit polling resilient to transient timeouts and HTTP 429 rate limits.
   - Made `API_BASE_URL` configurable via `os.environ.get("API_BASE_URL", ...)`.

4. **Commit `389fb28`**: `perf(vectorstore): switch QdrantClient to in-memory to prevent disk lock contention in concurrent background tasks`
   - Converted QdrantClient to `location=":memory:"` to eliminate SQLite file locking (`./local_qdrant_storage/storage.lock`) and thread contention.

---

## 4. Local Verification & Proof of Success

We executed a full end-to-end run locally with `main.py` on *"Future of Artificial Intelligence"*:
- **Search:** Scraped 15 authoritative web sources via Tavily.
- **Retriever:** Embedded and indexed 33 document chunks into Qdrant using FastEmbed, applied Trust-Tier domain multipliers (ranking `who.int`, `ieee.org`, and `nature.com` at the top), and extracted the top 5 chunks.
- **Critic:** Evaluated context quality with RAGAS criteria and **approved with a score of 0.85 / 1.0**.
- **Writer:** Synthesized a multi-section Markdown research report with inline domain citations, H2/H3 headers, and an Executive Synthesis.

---

## 5. Next Steps for Tomorrow

1. **Address Render Free Tier Memory (512MB RAM)**:
   - When FastEmbed (`bge-small-en-v1.5`) runs alongside Uvicorn, FastAPI, and LangGraph on Render Free, memory can spike near the 512MB limit, causing Render to restart the web service.
   - **Options to finalize tomorrow:**
     - **Option A (Recommended for 100% Free Tier):** Integrate a cloud embedding provider (like Google's `text-embedding-004` or an external lightweight API) with fallback, so Render never has to load ONNX model weights into memory.
     - **Option B:** Upgrade Render from Free to Starter ($7/mo with 2GB RAM / 1 vCPU), allowing FastEmbed and Qdrant to run in-memory without resource constraints.

2. **Database Orphaned Run Cleanup**:
   - In `app/api/main.py`, add a startup event that checks for any records with `status="running"` from before the server restart and marks them as `status="failed"` (*"Process restarted mid-execution"*). This prevents the UI from polling stale orphaned runs indefinitely.

3. **Full Streamlit End-to-End Test**:
   - Run Streamlit locally (`streamlit run app/frontend/app.py`) and on cloud (`https://researchgpt-ai.streamlit.app/`) to submit a live prompt and verify UI completion.
