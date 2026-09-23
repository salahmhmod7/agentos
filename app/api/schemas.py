"""Pydantic schemas for the API layer."""

from datetime import datetime

from pydantic import BaseModel, Field


# ---------- Requests ----------

class ChatRequest(BaseModel):
    """Send a message to the agent."""

    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: int | None = Field(
        default=None,
        description="Existing conversation id. If null, a new one is created.",
    )


# ---------- Responses ----------

class ToolCallOut(BaseModel):
    iteration: int
    tool_name: str
    arguments: dict
    result_preview: str
    error: str | None = None


class ChatResponse(BaseModel):
    conversation_id: int
    run_id: int | None = None
    status: str
    iterations: int
    final_answer: str | None
    tool_calls: list[ToolCallOut]


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime | None
    run_count: int


class RunSummary(BaseModel):
    id: int
    user_message: str
    final_answer: str | None
    status: str
    iterations: int
    tool_call_count: int
    model: str
    started_at: datetime | None
    ended_at: datetime | None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model: str
    database: str
class StreamChatRequest(BaseModel):
    """Query parameters for the streaming chat endpoint."""

    message: str = Field(..., min_length=1, max_length=5000)
    conversation_id: int | None = None
class DocumentInfo(BaseModel):
    id: int
    name: str
    extension: str | None = None
    num_pages: int = 1
    num_chunks: int = 0
    created_at: str | None = None


class ChunkPreview(BaseModel):
    id: int
    chunk_index: int
    text: str
    metadata: dict = {}


class DocumentDetail(DocumentInfo):
    chunks: list[ChunkPreview] = []


class StatsResponse(BaseModel):
    conversations: int
    agent_runs: int
    tool_calls: int
    documents: int
    chunks: int
    provider: str
    model: str