"""
Database Connection Management for DeepTrace Phase 5

Provides the SQLAlchemy synchronous engine and sessionmaker.
Uses psycopg2 driver for robust, synchronous database access.
"""

import os
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Ensure environment variables are loaded for DB connections
load_dotenv()

# Safely build connection URL using SQLAlchemy URL.create()
# This guarantees that special characters in passwords (e.g., @, /, #) are properly URL-encoded.
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg2",
    username=os.environ.get("POSTGRES_USER", "deeptrace"),
    password=os.environ.get("POSTGRES_PASSWORD", "deeptrace_password"),
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=os.environ.get("POSTGRES_PORT", "5432"),
    database=os.environ.get("POSTGRES_DB", "deeptrace_db")
)

# Create the SQLAlchemy engine
# pool_pre_ping=True gracefully handles stale/dropped connections by verifying them before checkout
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Configured "Session" factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency generator that yields a database session.
    Safe to use within synchronous LangGraph threads or FastAPI dependencies.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
