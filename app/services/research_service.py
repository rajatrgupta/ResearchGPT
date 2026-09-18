import uuid
from app.graph.workflow import build_graph
from app.database.connection import SessionLocal
from app.database.models import ResearchRun

def execute_research_run(run_id: str, query: str):
    """
    Executes the LangGraph research workflow synchronously.
    Updates the PostgreSQL database with stage transitions and the final report.
    """
    db = SessionLocal()
    try:
        # 1. Fetch the pending run from DB
        run_record = db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()
        if not run_record:
            print(f"[Service] Error: Run ID {run_id} not found in DB.")
            return

        # Update status to running
        run_record.status = "running"
        db.commit()
        print(f"[Service] Run {run_id} started processing.")

        # 2. Build Graph and Initial State
        graph = build_graph()
        initial_state = {
            "query": query,
            "run_id": run_id, # Crucial for Qdrant isolation
            "sub_questions": [],
            "search_results": [],
            "retrieved_documents": [],
            "sources": [],
            "report": "",
            "quality_score": 0.0,
            "iteration_count": 0,
            "status": "initialized",
            "errors": [],
            "critic_feedback": "",
            "is_valid": False,
            "retry_queries": []
        }

        current_state = initial_state

        # 3. Execute the Graph and track progress
        # Using .stream() allows us to intercept the state after every node executes
        for output in graph.stream(initial_state):
            for node_name, state_update in output.items():
                current_state.update(state_update)

                # Update database with current execution stage
                run_record.current_stage = node_name

                # Save intermediate outputs for future Phase 8 RAGAS evaluation
                if node_name == "planner":
                    run_record.sub_questions = current_state.get("sub_questions", [])
                elif node_name == "retriever":
                    run_record.retrieved_documents = current_state.get("retrieved_documents", [])

                db.commit()
                print(f"[Service] Run {run_id} completed node: {node_name}")

        # 4. Finalize the run
        run_record.status = "completed"
        run_record.current_stage = "completed"
        run_record.final_report = current_state.get("report", "No report generated.")
        db.commit()
        print(f"[Service] Run {run_id} completed successfully.")

    except Exception as e:
        # Handle catastrophic failures gracefully
        print(f"[Service] Run {run_id} failed: {str(e)}")
        # Rollback any pending failed transactions to prevent PendingRollbackError
        db.rollback() 
        run_record = db.query(ResearchRun).filter(ResearchRun.run_id == run_id).first()
        if run_record:
            run_record.status = "failed"
            run_record.error_message = str(e)
            try:
                db.commit()
            except Exception as inner_e:
                print(f"[Service] Fatal: Could not update DB with failed status: {str(inner_e)}")
    finally:
        db.close()
