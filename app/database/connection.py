"""
Database Connection Management for ResearchGPT

Provides the SQLAlchemy synchronous engine and sessionmaker.
Uses psycopg2 driver for robust, synchronous database access.
"""

import os
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Ensure environment variables are loaded for DB connections
load_dotenv()

# Check if Render's DATABASE_URL is present
db_url_env = os.environ.get("DATABASE_URL")

if db_url_env:
    # Render provides 'postgres://', but SQLAlchemy+psycopg2 needs 'postgresql+psycopg2://'
    DATABASE_URL = db_url_env.replace("postgres://", "postgresql+psycopg2://")
else:
    # Local development fallback
    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ.get("POSTGRES_USER", "deeptrace"),
        password=os.environ.get("POSTGRES_PASSWORD", "deeptrace_password"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        database=os.environ.get("POSTGRES_DB", "deeptrace_db")
    )

# Create the SQLAlchemy engine
# pool_pre_ping=True gracefully handles stale/dropped connections
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