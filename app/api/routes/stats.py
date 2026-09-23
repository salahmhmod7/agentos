"""Stats endpoint — aggregated counts for the dashboard."""

import sqlite3

from fastapi import APIRouter

from app.agent.rag.store import DB_PATH as RAG_DB_PATH
from app.api.schemas import StatsResponse
from app.core.config import settings


router = APIRouter(tags=["system"])


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Return counts across conversations, runs, tools, and documents."""
    conv = 0
    runs = 0
    tools = 0
    docs = 0
    chunks = 0

    try:
        conn = sqlite3.connect(RAG_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            conv = conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()["c"]
            runs = conn.execute("SELECT COUNT(*) AS c FROM agent_runs").fetchone()["c"]
            tools = conn.execute("SELECT COUNT(*) AS c FROM tool_calls").fetchone()["c"]
            docs = conn.execute("SELECT COUNT(*) AS c FROM rag_documents").fetchone()["c"]
            chunks = conn.execute("SELECT COUNT(*) AS c FROM document_chunks").fetchone()["c"]
        finally:
            conn.close()
    except Exception:
        pass

    return StatsResponse(
        conversations=conv,
        agent_runs=runs,
        tool_calls=tools,
        documents=docs,
        chunks=chunks,
        provider=settings.llm_provider,
        model=settings.groq_model if settings.llm_provider == "groq" else settings.ollama_model,
    )
