# Phase 5: Backend & Database Integration (Completed)

## 1. Architectural Shift
We successfully transitioned DeepTrace from a transient CLI-based script into a persistent, asynchronous REST API backend. 
- **Previous State:** CLI execution (`main.py`) blocking the terminal, ephemeral state, no persistence across runs.
- **Current State:** FastAPI-driven backend with background task execution, PostgreSQL-backed state persistence, and a decoupled service layer.

## 2. Phase 5A: Database Foundation
**Goal:** Establish persistent storage for research runs.
- **Infrastructure:** Deployed PostgreSQL via Docker (`docker-compose.yml`) exposing port 5432.
- **ORM & Schema:** Integrated SQLAlchemy. Created the `ResearchRun` model (`app/database/models.py`) with fields: `run_id` (UUID), `query`, `status`, `current_stage`, `sub_questions`, `retrieved_documents`, `final_report`, and `error_message`.
- **Migrations:** Initialized Alembic. Generated and applied the initial migration (`706f944a464f_create_research_runs_table.py`) to safely create the `research_runs` table.
- **Validation:** Wrote and executed `test_db.py` to validate CRUD (Create, Read, Update, Delete) operations directly against the PostgreSQL container. 

## 3. Phase 5B: FastAPI Integration & Service Layer
**Goal:** Expose the LangGraph execution through REST APIs without blocking the client.
- **Service Layer Extraction:** Created `app/services/research_service.py`. Moved the LangGraph initialization, state tracking, and streaming loop from `main.py` into `execute_research_run()`.
- **Database Interceptors:** Injected SQLAlchemy `db.commit()` calls inside the `.stream()` loop to update the `current_stage` in PostgreSQL dynamically as LangGraph agents execute (e.g., planner -> search -> retriever -> critic -> writer).
- **FastAPI Application:** Created `app/api/main.py` serving on Uvicorn.
  - Resolved environment variable initialization by injecting `dotenv` loader at the top of the API entry point.
- **API Endpoints:**
  - `POST /api/v1/research`: Accepts a JSON payload (`query`), generates a UUID `run_id`, initializes a `pending` DB record, and delegates the LangGraph execution to FastAPI's `BackgroundTasks`. Returns `202 Accepted` instantly.
  - `GET /api/v1/research/{run_id}`: Polling endpoint for the frontend to fetch the real-time execution status, current stage, and the final Markdown report from the database.

## 4. Run Isolation (Qdrant & PostgreSQL)
- **Technical Debt Resolved:** In the new API structure, the temporary `reset_collection()` call for Qdrant (previously in `main.py`) was entirely bypassed. 
- **Current Isolation:** Every search and retrieval cycle is now strictly isolated via the UUID `run_id`. Qdrant filters vector retrieval based on this `run_id`, preventing data contamination across concurrent API requests.
