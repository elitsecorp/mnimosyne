"""SQLAlchemy engine, session management, and database initialization."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from mnemosyne.config import settings
from mnemosyne.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """Return the shared SQLAlchemy engine (lazy singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        _enable_wal(_engine)
    return _engine


def _enable_wal(engine) -> None:
    """Enable WAL mode for better concurrent read performance."""
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_session_factory():
    """Return a session factory bound to the engine."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_db():
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables and run migrations for existing databases."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _migrate_add_session_id(engine)
    logger.info("Database initialized.")


def _migrate_add_session_id(engine) -> None:
    """Add session_id to messages table if missing (migration for existing DBs)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "chat_sessions" not in tables:
        logger.info("Migrating: creating chat_sessions table")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT 'New Chat', created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("INSERT INTO chat_sessions (title, created_at, updated_at) VALUES ('Default', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
    columns = [col["name"] for col in inspector.get_columns("messages")]
    if "session_id" not in columns:
        logger.info("Migrating: adding session_id to messages table")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE messages ADD COLUMN session_id INTEGER REFERENCES chat_sessions(id)"))
            conn.execute(text("UPDATE messages SET session_id = (SELECT id FROM chat_sessions LIMIT 1) WHERE session_id IS NULL"))
        logger.info("Migration complete: session_id added")


def load_schema_sql() -> str:
    """Load the reference schema.sql file."""
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text()
    return ""
