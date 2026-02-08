import logging
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import text

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.sessions import router as sessions_router
from app.db.database import Base, DATABASE_BACKEND, get_engine, log_db_info

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Atlantis AI Chatbot")

# Avoid repeated table creation in the same process.
_tables_initialized = False

app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(sessions_router)


@app.on_event("startup")
def on_startup() -> None:
    # Startup safety checks and strict DB connection validation.
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    logger.info(
        "LLM provider configured",
        extra={"endpoint": "startup", "error_type": "none", "provider": llm_provider},
    )
    log_db_info()
    engine = get_engine()

    if DATABASE_BACKEND == "sqlite":
        Base.metadata.create_all(bind=engine)
        logger.info(
            "Database connected successfully",
            extra={"endpoint": "startup", "error_type": "none"},
        )
        return

    # Postgres: retry up to 5 times with clear logging.
    for attempt in range(1, 6):
        try:
            with engine.begin() as connection:
                connection.execute(text("SELECT 1"))
                # Ensure pgvector extension exists before creating tables that use it.
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(bind=engine)
            logger.info(
                "Database connected successfully",
                extra={"endpoint": "startup", "error_type": "none"},
            )
            return
        except Exception as exc:
            logger.error(
                "Database connection failed (attempt %s/5): %s",
                attempt,
                exc,
            )
            if attempt == 5:
                if "password authentication failed" in str(exc):
                    raise RuntimeError(
                        "PostgreSQL authentication failed. "
                        "Check password, pgAdmin role, and ensure Postgres service is running."
                    ) from exc
                raise RuntimeError(
                    "PostgreSQL connection failed. Check DATABASE_URL and Postgres service."
                ) from exc
            time.sleep(2)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
