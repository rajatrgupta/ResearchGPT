# DeepTrace Context and Memory: Phase 5A

## SECTION A — WHY PHASE 5 EXISTS
**Before Phase 5**: DeepTrace was an ephemeral CLI tool. You would run `python main.py`, wait for the agents to do their research, print the report, and then the process would end. 

**The Problem**:
- No persistent history (the final report and agent decisions vanish when the terminal closes).
- No API (external applications cannot trigger a research job).
- Difficult UI integration (a frontend cannot cleanly talk to a CLI script).
- State disappears entirely when the process exits.

**Phase 5 Goal**: Turn DeepTrace from a stateless CLI research script into a persistent, API-driven application backend. We are building the foundational infrastructure to support real users, UI dashboards, and historical evaluation.

## SECTION B — PHASE 5 BIG PICTURE
**Target Architecture**:
Client (cURL/Frontend) → HTTP `POST` to FastAPI → Creates Row in PostgreSQL → Triggers LangGraph in a Background Thread → Updates PostgreSQL Status → Client polls HTTP `GET` to FastAPI → Client receives Final Report.

**Responsibilities**:
- **FastAPI**: Handles HTTP requests, polling, and JSON responses.
- **PostgreSQL**: Stores the persistent, client-facing application state (run IDs, status, queries, final reports).
- **LangGraph**: The actual research intelligence engine.
- **Qdrant**: Stores vector embeddings and chunked context data for semantic search.

## SECTION C — PHASE 5 ROADMAP
The engineering plan for Phase 5 is divided into strict milestones:
- **Phase 5A — Database Foundation**: Setup Postgres, SQLAlchemy models, and Alembic migrations. (Goal: Prove DB connectivity without breaking existing CLI behavior).
- **Phase 5B — Research Service**: Refactor `main.py` into a callable service, remove Qdrant collection resets, and implement run-based isolation.
- **Phase 5C — FastAPI Endpoints**: Build the actual HTTP routers (`POST /research`, `GET /research/{id}`).
- **Phase 5D — Integration Testing**: End-to-end testing from HTTP request to database persistence.
- **Phase 5E — Documentation**: Final wrap-up.

## SECTION D — WHAT WE HAVE ALREADY DONE IN 5A
We have successfully implemented the codebase for Phase 5A:
- **`docker/docker-compose.yml`**: Configured a local `postgres:15-alpine` container with a persistent volume to ensure reproducibility.
- **`requirements.txt`**: Added `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.0`, and `alembic>=1.13.0`.
- **`.env.example` & `.env`**: Added safe placeholders and local credentials for database connections.
- **`app/database/connection.py`**: Created the SQLAlchemy `engine` and `SessionLocal` factory using `URL.create()` to safely encode database passwords with special characters.
- **`app/database/models.py`**: Defined the `ResearchRun` schema.
- **`alembic/` & `alembic.ini`**: Initialized the migration environment.
- **`alembic/env.py`**: Configured to dynamically import `app.database.models.Base` and `DATABASE_URL` for automatic schema detection.

**ResearchRun Schema Explained**:
- `run_id` (UUID): Primary key, links HTTP requests, DB rows, and Qdrant vectors.
- `query` (Text): The user's original research topic.
- `status` (String): Tracks high-level lifecycle (`pending`, `running`, `completed`, `failed`).
- `current_stage` (String): Tracks fine-grained LangGraph node progress.
- `sub_questions`, `retrieved_documents` (JSONB): Preserves intermediate agent decisions for future evaluation (Phase 8).
- `final_report` (Text): The ultimate markdown deliverable.
- `error_message` (Text): Captures exceptions if the pipeline crashes.
- `created_at`, `completed_at` (DateTime): Metrics for duration tracking.

## SECTION E — IMPORTANT ARCHITECTURAL DECISIONS
- **PostgreSQL**: Chosen for robust JSONB support and transactional integrity.
- **SQLAlchemy 2.0 + psycopg2 (Synchronous)**: We intentionally avoided `asyncpg` because our LangGraph agent code (LLM calls, Tavily searches) is heavily synchronous. Mixing async DB drivers with sync execution threads creates immense session-management complexity.
- **Alembic**: Used for clean, version-controlled schema evolution instead of relying on `Base.metadata.create_all()`.
- **Why NOT Celery/Redis?**: Dedicated worker queues are production-standard but represent massive over-engineering for this phase. We are prioritizing lightweight `asyncio.to_thread` execution inside FastAPI first.

## SECTION F — DATABASE VS QDRANT
- **PostgreSQL**: Stores the *metadata* and *business state* (queries, run status, final reports).
- **Qdrant**: Stores the *mathematical intelligence* (high-dimensional embeddings, semantic chunks).
We do NOT duplicate raw scraped web snippets into PostgreSQL. Postgres only stores the highly curated Top-5 documents that actually reach the Writer. 

*Crucial Note for Phase 5B*: Because we are moving to concurrent API execution, we MUST stop deleting the entire Qdrant collection on every run. Phase 5B will remove the dangerous `reset_collection()` call and rely entirely on the `run_id` isolation introduced in Phase 4B.

## SECTION G — PHASE 5A CURRENT BLOCKER
**Current Status**: Phase 5A execution is currently **BLOCKED**.
- **What happened**: The `docker-compose up -d` command failed with the error: `unable to get image 'postgres:15-alpine': failed to connect to the docker API`.
- **Why**: The host machine (AMD Ryzen 5 3550H) has hardware virtualization (AMD SVM) disabled in the BIOS. As a result, the Docker Desktop daemon cannot start the Linux engine required to run the PostgreSQL container.
- **Assessment**: This is entirely an environment/BIOS hardware limitation. It is **NOT** a DeepTrace application-code failure or bug.

## SECTION H — EXACT RESUME PLAN
To unblock Phase 5A, the following steps must be taken manually on the host machine:
1. Enter BIOS/UEFI on reboot.
2. Enable AMD SVM / Hardware Virtualization.
3. Save and restart Windows.
4. Verify Task Manager -> Performance -> CPU says "Virtualization: Enabled".
5. Start Docker Desktop and wait for the daemon to turn green/ready.
6. Return to this CLI and approve resumption of Phase 5A.
7. The agent will then start the PostgreSQL container (`docker-compose up -d`).
8. Generate the initial Alembic migration.
9. Inspect the migration to ensure tables are detected.
10. Run `alembic upgrade head`.
11. Run a lightweight DB insertion/read validation script.
12. Confirm Phase 5A is complete before touching Phase 5B.

## SECTION I — PHASE 5B PLAN
Once the database foundation is verified, Phase 5B will decouple the research logic from the CLI script.
- **Refactor `main.py`**: Convert the raw execution script into a callable Python service function.
- **Postgres Updates**: Inject database session callbacks into the LangGraph state execution to update `status` and `current_stage` in real-time.
- **Qdrant Fix**: Remove the `client.delete_collection()` call. We will rely purely on `run_id` FieldConditions to isolate data, allowing multiple API users to search simultaneously without wiping each other's vector memory.

## SECTION J — PHASE 5C PLAN
We will expose the backend via two FastAPI REST endpoints:
- `POST /api/v1/research`: Accepts a Pydantic validated query, creates a DB row, fires the background task, and returns `202 Accepted` with a `run_id`.
- `GET /api/v1/research/{run_id}`: Polling endpoint for the frontend UI to fetch the `status` and `final_report`. It masks internal implementation details and handles `404 Not Found` securely.

## SECTION K — LONG-RUNNING EXECUTION
**Chosen Architecture**: `FastAPI BackgroundTasks` + `asyncio.to_thread()` + Synchronous LangGraph.
**Why**: DeepTrace research takes 30–120+ seconds. Running this synchronously would choke the ASGI event loop and crash the web server. `asyncio.to_thread` pushes the blocking network calls into a background thread pool, keeping the API responsive to incoming polling requests. Celery/RQ is intentionally deferred to Phase 10 to maintain development velocity and keep the infrastructure footprint small.

## SECTION L — PHASE 5D TESTING
Validation will involve end-to-end API testing without a UI:
- Issue a `POST` request.
- Receive a `run_id` and `pending` status.
- Poll the `GET` endpoint and observe `running` and stage transitions.
- Verify completion and persistence of the markdown report.
- Verify failed state handling if an API key is revoked mid-run.
- Verify Qdrant vectors are safely isolated.

## SECTION M — FUTURE ROADMAP
The architecture laid down in Phase 5 enables the rest of the roadmap:
- **Phase 6 — LangSmith Tracing**: Moves observability to the cloud (API requests hide console logs, making traces mandatory before building the UI).
- **Phase 7 — Streamlit Frontend**: A simple UI client that consumes our new robust Phase 5 FastAPI endpoints.
- **Phase 8 — RAGAS Evaluation**: Uses the persistent PostgreSQL history to score faithfulness and relevance over time.
- **Phase 9 — Hybrid Retrieval / BM25**: Refines search quality.
- **Phase 10 — Cross-Session Memory & Productionization**: Celery workers, user accounts, and conversational memory checkpoints.

## SECTION N — CURRENT STATUS
Phase 1 — COMPLETE
Phase 2 — COMPLETE
Phase 3A — COMPLETE
Phase 3B — COMPLETE
Phase 4A — COMPLETE
Phase 4B — COMPLETE WITH MINOR TECH DEBT
Phase 5A — IN PROGRESS / BLOCKED BY DOCKER VIRTUALIZATION

**NEXT ACTION**:
Enable AMD SVM virtualization in BIOS, restart Windows, verify Docker Desktop, then resume Phase 5A from PostgreSQL container startup.
