import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.database import get_db
from app.db.models import ChatMessage, DocumentEmbedding
from app.db.sessions import get_or_create_session
from app.services.rag import generate_response


router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: UUID | None = None
    message: str = Field(..., min_length=1)


@router.post("")
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> dict:
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # Create a new session when none is provided; otherwise reuse it.
    chat_session = get_or_create_session(db, request.session_id)

    # RAG works best when it has the full conversation context for grounding.
    history_rows = db.execute(
        select(ChatMessage.role, ChatMessage.content)
        .where(ChatMessage.session_id == chat_session.id)
        .order_by(ChatMessage.created_at)
    ).all()

    has_embeddings = (
        db.execute(
            select(DocumentEmbedding.id)
            .where(DocumentEmbedding.session_id == chat_session.id)
            .where(DocumentEmbedding.embedding.is_not(None))
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )

    if not has_embeddings:
        reply = (
            "Documents uploaded successfully, but semantic search is disabled due to "
            "missing embeddings."
        )
    else:
        try:
            reply = generate_response(
                db, chat_session.id, message, history_override=history_rows
            )
        except Exception as exc:
            error_text = str(exc).lower()
            if any(
                token in error_text
                for token in (
                    "insufficient_quota",
                    "429",
                    "connecterror",
                    "connection",
                    "refused",
                    "unavailable",
                    "timeout",
                )
            ):
                logger.warning(
                    "LLM temporarily unavailable",
                    extra={
                        "session_id": str(chat_session.id),
                        "endpoint": "/api/chat",
                        "error_type": "llm_unavailable",
                    },
                )
                reply = "LLM temporarily unavailable. Please try again later."
            else:
                logger.exception(
                    "Chat generation failed",
                    extra={
                        "session_id": str(chat_session.id),
                        "endpoint": "/api/chat",
                        "error_type": type(exc).__name__,
                    },
                )
                raise HTTPException(
                    status_code=502, detail="LLM response generation failed"
                ) from exc

    db.add(ChatMessage(session_id=chat_session.id, role="user", content=message))
    db.add(
        ChatMessage(session_id=chat_session.id, role="assistant", content=reply)
    )
    db.commit()

    return {"session_id": str(chat_session.id), "response": reply}
