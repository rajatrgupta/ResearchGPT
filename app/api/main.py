from dotenv import load_dotenv
load_dotenv()

import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import ResearchRun
from app.services.research_service import execute_research_run

app = FastAPI(
    title="ResearchGPT API",
    description="Autonomous Multi-Agent Research System API",
    version="1.0.0"
)

@app.on_event("startup")
def cleanup_orphaned_runs():
    """
    On server startup, find any runs left in 'running' or 'pending' state
    from a previous server crash/restart and mark them as failed.
    Prevents the UI from polling stale orphaned runs indefinitely.
    """
    db = SessionLocal()
    try:
        orphaned = db.query(ResearchRun).filter(ResearchRun.status.in_(["pending", "running"])).all()
        if orphaned:
            print(f"[Startup] Found {len(orphaned)} orphaned runs from previous server session. Marking as failed...")
            for run in orphaned:
                run.status = "failed"
                run.error_message = "Server restarted mid-execution. Please trigger a new research run."
            db.commit()
            print("[Startup] Orphaned runs cleaned up successfully.")
    except Exception as e:
        print(f"[Startup] Warning: Failed to clean up orphaned runs: {e}")
        db.rollback()
    finally:
        db.close()

@app.get("/")
@app.get("/health")
def health_check():
    """Liveliness probe for Render and client health checks."""
    return {
        "status": "healthy",
        "service": "ResearchGPT API",
        "version": "1.0.0"
    }

@app.get("/api/v1/debug-system")
def debug_system():
    """Safe diagnostic endpoint to verify model and vectorstore readiness without exposing secrets."""
    import os
    from app.vectorstore.qdrant_store import get_embedding_model
    emb = get_embedding_model()
    return {
        "embedding_mode": emb._mode,
        "vector_size": emb.vector_size,
        "has_google_key": bool(os.environ.get("GOOGLE_API_KEYS") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "has_tavily_key": bool(os.environ.get("TAVILY_API_KEY")),
        "has_openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "has_database_url": bool(os.environ.get("DATABASE_URL"))
    }

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ResearchRequest(BaseModel):
    query: str

@app.post("/api/v1/research", status_code=202)
def start_research(request: ResearchRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Accepts a research query, creates a run_id, saves a pending status to DB,
    and triggers the LangGraph execution in the background.
    """
    run_id = str(uuid.uuid4())
    
    new_run = ResearchRun(
        run_id=run_id,
        query=request.query,
        status="pending",
        current_stage="initialized"
    )
    db.add(new_run)
    db.commit()
    
    background_tasks.add_task(execute_research_run, run_id, request.query)
    
    return {
        "message": "Research task accepted.",
        "run_id": run_id,
        "status": "pending"
    }

@app.get("/api/v1/research/{run_id}")
def get_research_status(run_id: str, db: Session = Depends(get_db)):
    """
    Polling endpoint for the frontend to check research progress and fetch the final report.
    """
    run_record = db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()
    
    if not run_record:
        raise HTTPException(status_code=404, detail="Run ID not found.")
        
    return {
        "run_id": run_record.run_id,
        "query": run_record.query,
        "status": run_record.status,
        "current_stage": run_record.current_stage,
        "final_report": run_record.final_report,
        "error_message": run_record.error_message
    }
