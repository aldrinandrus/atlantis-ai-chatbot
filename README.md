# Atlantis AI Chatbot

## Project overview

FastAPI backend for a session-based AI chatbot with document upload, pgvector
retrieval, and LangChain-powered responses.

## API endpoints

- `POST /api/sessions` - create a new chat session
- `POST /api/documents/upload` - upload PDF or text file for a session
- `POST /api/chat` - send a chat message and receive a response
- `GET /health` - health check

## Environment variables

Required:
- `DATABASE_URL` - PostgreSQL connection string

LLM:
- `LLM_PROVIDER` - `openai`, `gemini`, or `anthropic`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-1.5-flash`)
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL` (default: `claude-3-5-sonnet-latest`)
- `EMBEDDING_PROVIDER` (default: `openai`)

## Database schema

- `sessions` - UUID primary key and `created_at`
- `chat_messages` - chat history (`session_id`, `role`, `content`, `created_at`)
- `document_embeddings` - chunked text with vector embeddings and optional JSON metadata

## PostgreSQL setup

- Requirement: PostgreSQL 15+ (with `pgvector` support).
- Check that PostgreSQL is running (Windows PowerShell):
```
Get-Service postgresql*
```
- Create the database:

```sql
CREATE DATABASE chatbot;
```

- Set or update the postgres user password (psql):
```
psql -U postgres
ALTER USER postgres WITH PASSWORD 'your_password_here';
```

- Enable the extension in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

- Update `DATABASE_URL` to match your credentials:
```
DATABASE_URL=postgresql://postgres:your_password_here@localhost:5432/chatbot
```

## Windows Setup – PostgreSQL

- Install PostgreSQL from: https://www.postgresql.org/download/windows/
- Set the `postgres` password using pgAdmin:
  - Open pgAdmin, right-click `postgres` role → Properties → Definition → Password.
- Create a new role (recommended) in pgAdmin:
  - Login Roles → Create → Login/Group Role → set username/password → grant privileges.
  - Update `DATABASE_URL` to use the new role.
- Fix `psql` not recognized:
  - Add `C:\Program Files\PostgreSQL\<version>\bin` to your PATH, or use pgAdmin's query tool.
- Example `DATABASE_URL` for local dev:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/chatbot
```
- SQLite fallback exists to keep local development unblocked if Postgres auth fails.
- To force SQLite locally, set:
```
DATABASE_BACKEND=sqlite
```

## Local setup

1) Create and activate a virtual environment.
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Set environment variables (see `.env.example`).
4) Start the server:
```
uvicorn app.main:app --reload
```

## Docker setup

Prerequisites:
- Docker Desktop
- Docker Compose (included with Docker Desktop)

Steps:
1) Build and start services:
```
docker-compose up --build
```
2) Open Swagger UI:
```
http://localhost:8000/docs
```
3) The database and tables are auto-created on startup using SQLAlchemy.

## Troubleshooting

- Postgres version mismatch: If you see errors about data files created by a different
  PostgreSQL version, your existing Docker volume was initialized with another version.
- Fix: stop containers and remove volumes to recreate with the new version:
```
docker-compose down -v
```
- Why this is required: Postgres data directories are not always forward-compatible
  across major versions, so the volume must be recreated when the image version changes.

## Example curl requests

Create a session:
```
curl -X POST http://localhost:8000/api/sessions
```

Upload a document:
```
curl -X POST http://localhost:8000/api/documents/upload \
  -F "session_id=<SESSION_ID>" \
  -F "file=@/path/to/document.pdf"
```

Chat with the assistant:
```
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<SESSION_ID>","message":"Summarize the document."}'
```

Note for Windows PowerShell:
- `curl` is an alias for `Invoke-WebRequest`, so `-X POST` is invalid.
- Use:
```
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8000/api/sessions
```
- Or call the real curl:
```
curl.exe -X POST http://127.0.0.1:8000/api/sessions
```
- Recommended for testing: Swagger UI at `http://localhost:8000/docs`.

## Testing with Postman or curl

Upload file:
- `POST /api/documents/upload`

Chat:
- `POST /api/chat`

Verify:
- Context is used (ask a question only answerable from the uploaded file).
- Previous messages are remembered (ask a follow-up that depends on prior turns).

## Design Decisions

- Session-scoped storage keeps chat history and document context isolated per user.
- pgvector powers semantic search while staying in Postgres for simpler operations.
- PDF parsing is defensive to avoid crashes on malformed or encrypted files.

## Failure Handling Strategy

- OpenAI quota/rate-limit errors return a successful response with a warning note.
- Uploads succeed even if embeddings are skipped; text is still stored.
- Startup checks log DB and pgvector readiness without crashing the server.

## How to Run Locally (Docker & Non-Docker)

- Non-Docker: install dependencies, set `DATABASE_URL`, run `uvicorn app.main:app --reload`.
- Docker: `docker-compose up --build` then open `http://localhost:8000/docs`.
