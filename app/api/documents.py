import logging
import os
import traceback
from io import BytesIO
from uuid import UUID

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import DocumentEmbedding, Session as ChatSession


router = APIRouter(prefix="/api/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_document(
    session_id: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    try:
        logger.info(
            "Upload request received",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )

        load_dotenv()
        gemini_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMINI_API_KEY_1")
            or os.getenv("GEMINI_API_KEY_2")
        )
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        # Validate inputs explicitly.
        if file is None:
            raise HTTPException(status_code=400, detail="File is required")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        try:
            session_uuid = UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid session_id") from exc
        if file.content_type not in {"application/pdf", "text/plain"}:
            raise HTTPException(status_code=400, detail="Only PDF or text files supported")

        # Validate session exists before processing.
        existing_session = db.execute(
            select(ChatSession).where(ChatSession.id == session_uuid)
        ).scalar_one_or_none()
        if not existing_session:
            raise HTTPException(status_code=400, detail="Invalid session_id")

        logger.info(
            "Before saving file",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )
        os.makedirs("uploads", exist_ok=True)
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        upload_path = os.path.join("uploads", f"{session_uuid}_{file.filename or 'upload'}")
        with open(upload_path, "wb") as handle:
            handle.write(content)
        logger.info(
            "After saving file",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )

        extracted_pages: list[str] = []
        if file.content_type == "application/pdf":
            logger.info(
                "Before PDF parsing",
                extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
            )
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise HTTPException(
                        status_code=400, detail="Encrypted PDF is not supported"
                    ) from exc
            for page in reader.pages:
                page_text = page.extract_text() if page else None
                if page_text:
                    extracted_pages.append(page_text)
            if not extracted_pages:
                raise HTTPException(status_code=400, detail="No readable text found in PDF")
            text = "\n".join(extracted_pages)
            logger.info(
                "After PDF parsing",
                extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
            )
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail="Text file must be UTF-8") from exc
            if not text.strip():
                raise HTTPException(status_code=400, detail="Text file is empty")
            extracted_pages = [text]

        # Split the extracted text into overlapping chunks for better retrieval.
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks created")

        logger.info(
            "Before embeddings creation",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )
        embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=gemini_key,
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
        )
        vectors = None
        try:
            vectors = embedding_model.embed_documents(chunks)
        except Exception as e:
            error_text = str(e).lower()
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
                    "Gemini temporarily unavailable",
                    extra={
                        "session_id": session_id,
                        "endpoint": "/api/documents/upload",
                        "error_type": "gemini_unavailable",
                    },
                )
                traceback.print_exc()
                vectors = None
            else:
                traceback.print_exc()
                logger.exception(
                    "Gemini embedding error",
                    extra={
                        "session_id": session_id,
                        "endpoint": "/api/documents/upload",
                        "error_type": "gemini_embedding",
                    },
                )
                raise HTTPException(
                    status_code=502, detail="Gemini embedding error"
                ) from e

        logger.info(
            "Before DB insert",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )
        if vectors is None:
            for chunk in chunks:
                db.add(
                    DocumentEmbedding(
                        session_id=session_uuid,
                        content=chunk,
                        embedding=None,
                        embeddings_status="pending",
                    )
                )
        else:
            for chunk, vector in zip(chunks, vectors, strict=False):
                db.add(
                    DocumentEmbedding(
                        session_id=session_uuid,
                        content=chunk,
                        embedding=vector,
                        embeddings_status="ready",
                    )
                )
        db.commit()
        if chunks:
            db.refresh(existing_session)
        logger.info(
            "After DB insert",
            extra={"session_id": session_id, "endpoint": "/api/documents/upload"},
        )

        response = {
            "status": "success",
            "session_id": str(session_uuid),
            "pages_processed": len(extracted_pages),
            "note": "Embeddings skipped if LLM unavailable",
        }
        if vectors is None:
            response["message"] = "LLM temporarily unavailable. Document uploaded successfully."
        return response
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.exception(
            "Unhandled error in upload_document",
            extra={
                "session_id": session_id,
                "endpoint": "/api/documents/upload",
                "error_type": type(e).__name__,
            },
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
