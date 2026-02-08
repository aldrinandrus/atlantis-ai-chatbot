from __future__ import annotations

from typing import Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, DocumentEmbedding
from app.services.embeddings import get_embeddings
from app.services.llm import get_llm
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def _format_history(messages: Iterable[tuple[str, str]]) -> str:
    lines = [f"{role}: {content}" for role, content in messages]
    return "\n".join(lines)


# Custom LangChain retriever backed by pgvector + SQLAlchemy.
class PgvectorRetriever(BaseRetriever):
    def __init__(self, db: Session, session_id: UUID, k: int = 4):
        super().__init__()
        self._db = db
        self._session_id = session_id
        self._k = k
        self._embeddings = get_embeddings()

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Embed the query and rank stored chunks by vector distance.
        query_vector = self._embeddings.embed_query(query)
        stmt = (
            select(DocumentEmbedding.content)
            .where(DocumentEmbedding.session_id == self._session_id)
            .order_by(DocumentEmbedding.embedding.cosine_distance(query_vector))
            .limit(self._k)
        )
        rows = self._db.execute(stmt).all()
        return [Document(page_content=row[0]) for row in rows]


def retrieve_context(
    db: Session, session_id: UUID, query: str, limit: int = 4
) -> list[str]:
    # Use a LangChain retriever abstraction to fetch relevant chunks.
    retriever = PgvectorRetriever(db=db, session_id=session_id, k=limit)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def generate_response(
    db: Session,
    session_id: UUID,
    query: str,
    context_limit: int = 4,
    history_limit: int = 20,
    history_override: Iterable[tuple[str, str]] | None = None,
) -> str:
    # Step 1: retrieve relevant document chunks via retriever.
    context_chunks = retrieve_context(db, session_id, query, limit=context_limit)

    if history_override is None:
        history_stmt = (
            select(ChatMessage.role, ChatMessage.content)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(history_limit)
        )
        history_rows = db.execute(history_stmt).all()
    else:
        history_rows = list(history_override)

    # Step 2: combine retrieved context with chat history.
    history = _format_history(history_rows)

    context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant context."
    history_text = history if history else "No prior messages."

    prompt = (
        "You are a helpful assistant. Use the provided context and chat history to "
        "answer the user's question. If the answer is not in the context, say you "
        "don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Chat history:\n{history_text}\n\n"
        f"User question:\n{query}\n"
    )

    # Step 3: pass combined context to the LLM.
    llm = get_llm()
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))
