import importlib.util
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables before reading DATABASE_URL.
load_dotenv()

DATABASE_BACKEND = (os.getenv("DATABASE_BACKEND") or "postgres").lower()
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_BACKEND not in {"postgres", "sqlite"}:
    raise RuntimeError("DATABASE_BACKEND must be 'postgres' or 'sqlite'")


def _build_postgres_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Expected format: postgresql://USER:PASSWORD@HOST:PORT/DBNAME"
        )
    try:
        url = make_url(DATABASE_URL)
    except Exception as exc:
        raise RuntimeError(
            "DATABASE_URL is invalid. Expected format: postgresql://USER:PASSWORD@HOST:PORT/DBNAME"
        ) from exc
    # Enforce Windows-friendly local host usage.
    if url.host in (None, "localhost"):
        url = url.set(host="127.0.0.1")
    return str(url)


def _build_sqlite_url() -> str:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'dev.db'}"


def _create_engine_for_backend(backend: str):
    if backend == "sqlite":
        return create_engine(_build_sqlite_url(), connect_args={"check_same_thread": False})
    return create_engine(_build_postgres_url())


# SQLAlchemy engine and session factory (lazy initialization).
engine = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

# Base class for ORM models.
Base = declarative_base()


def log_db_info() -> None:
    """Log DB connection details for Windows troubleshooting."""
    psycopg2_installed = importlib.util.find_spec("psycopg2") is not None
    if DATABASE_BACKEND == "sqlite":
        logging.info(
            "Database configuration",
            extra={
                "endpoint": "startup",
                "db_backend": "sqlite",
                "db_host": "local",
                "db_port": None,
                "db_user": None,
                "db_name": "data/dev.db",
                "psycopg2_installed": psycopg2_installed,
            },
        )
        return

    url = make_url(_build_postgres_url())
    logging.info(
        "Database configuration",
        extra={
            "endpoint": "startup",
            "db_backend": "postgres",
            "db_host": url.host,
            "db_port": url.port,
            "db_user": url.username,
            "db_name": url.database,
            "psycopg2_installed": psycopg2_installed,
        },
    )


def get_engine():
    """Return a lazily created engine for the configured backend."""
    global engine
    if engine is None:
        engine = _create_engine_for_backend(DATABASE_BACKEND)
        SessionLocal.configure(bind=engine)
    return engine


def get_db():
    """Provide a database session for request-scoped usage."""
    if engine is None:
        get_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
