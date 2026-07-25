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
    _migrate_add_is_owner(engine)
    _migrate_settings(engine)
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


def _migrate_add_is_owner(engine) -> None:
    """Add is_owner to relationships table if missing."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "relationships" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("relationships")]
        if "is_owner" not in columns:
            logger.info("Migrating: adding is_owner to relationships table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE relationships ADD COLUMN is_owner INTEGER DEFAULT 0"))
            logger.info("Migration complete: is_owner added")
        with engine.begin() as conn:
            if "valid_from" not in columns:
                conn.execute(text("ALTER TABLE relationships ADD COLUMN valid_from DATETIME"))
                logger.info("Added valid_from")
            if "valid_to" not in columns:
                conn.execute(text("ALTER TABLE relationships ADD COLUMN valid_to DATETIME"))
                logger.info("Added valid_to")
            if "last_seen" not in columns:
                conn.execute(text("ALTER TABLE relationships ADD COLUMN last_seen DATETIME"))
                logger.info("Added last_seen")


def _migrate_settings(engine) -> None:
    """Create settings table and seed default values."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "settings" not in inspector.get_table_names():
        logger.info("Migrating: creating settings table")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key VARCHAR(128) NOT NULL UNIQUE,
                    value TEXT NOT NULL DEFAULT '',
                    encrypted BOOLEAN DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
        logger.info("Migration complete: settings table created")
    else:
        columns = [col["name"] for col in inspector.get_columns("settings")]
        if "updated_at" not in columns:
            logger.info("Migrating: adding updated_at to settings table")
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE settings ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"))
            logger.info("Migration complete: updated_at added")

    factory = get_session_factory()
    db = factory()
    try:
        existing = {row[0] for row in db.execute(text("SELECT key FROM settings")).fetchall()}
        defaults = {
            "llm_provider": ("gemini", False),
            "llm_api_key": ("", True),
            "llm_model": ("gemini-2.0-flash", False),
            "ollama_url": ("http://localhost:11434", False),
            "max_context_length": ("8000", False),
            "max_vector_results": ("10", False),
            "max_entity_extraction": ("25", False),
            "max_graph_depth": ("3", False),
            "min_confidence": ("0.0", False),
            "min_similarity": ("0.6", False),
            "auto_consolidate_interval": ("5", False),
            "embedding_backend": ("local", False),
        }
        for key, (value, encrypted) in defaults.items():
            if key not in existing:
                db.execute(
                    text("INSERT INTO settings (key, value, encrypted, updated_at) VALUES (:key, :value, :encrypted, CURRENT_TIMESTAMP)"),
                    {"key": key, "value": value, "encrypted": encrypted},
                )
        db.commit()
        logger.info("Settings defaults seeded")
    finally:
        db.close()


def load_schema_sql() -> str:
    """Load the reference schema.sql file."""
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text()
    return ""
