"""Endpoints for inspecting conversations and runs."""

from fastapi import APIRouter, HTTPException

from app.agent import memory
from app.api.schemas import ConversationSummary, RunSummary
from app.db.base import session_scope
from app.db.models import Conversation


router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    """List all conversations (most recent first)."""
    with session_scope() as session:
        rows = (
            session.query(Conversation)
            .order_by(Conversation.id.desc())
            .limit(100)
            .all()
        )
        return [
            ConversationSummary(
                id=c.id,
                title=c.title,
                created_at=c.created_at,
                run_count=len(c.runs),
            )
            for c in rows
        ]


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int) -> dict:
    """Return full details of a conversation (runs + messages)."""
    history = memory.get_conversation_history(conversation_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = memory.load_conversation_messages(conversation_id, limit=200)
    history["messages"] = messages
    return history


@router.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: int) -> RunSummary:
    """Return a single agent run."""
    from app.db.models import AgentRun

    with session_scope() as session:
        run = session.get(AgentRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return RunSummary(
            id=run.id,
            user_message=run.user_message,
            final_answer=run.final_answer,
            status=run.status,
            iterations=run.iterations,
            tool_call_count=len(run.tool_calls),
            model=run.model,
            started_at=run.started_at,
            ended_at=run.ended_at,
        )