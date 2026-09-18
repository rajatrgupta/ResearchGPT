"""
SQLAlchemy ORM Models for DeepTrace Phase 5
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ResearchRun(Base):
    """
    Persistent application state for a single research execution.
    Provides the ground truth for frontend polling and future evaluation.
    """
    __tablename__ = "research_runs"

    # Core Identifier
    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Input
    query = Column(Text, nullable=False)
    
    # Lifecycle Tracking (pending, running, completed, failed)
    status = Column(String(50), nullable=False, default="pending")
    current_stage = Column(String(50), nullable=True)
    
    # Generated Artifacts (JSONB for native PostgreSQL indexing/querying)
    sub_questions = Column(JSONB, nullable=True)
    retrieved_documents = Column(JSONB, nullable=True)
    final_report = Column(Text, nullable=True)
    
    # Error Handling
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<ResearchRun(run_id={self.run_id}, status={self.status})>"
