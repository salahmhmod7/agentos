"""Memory layer — persists conversations, runs, and tool calls.

This is the ONLY module the rest of the app should use to touch the DB.
Everything goes through here so we can swap SQLite → PostgreSQL later.
"""

import json
from datetime import datetime, timezone

from app.agent.state import AgentState
from app.db.base import session_scope
from app.db.models import AgentRun, Conversation, Message, ToolCall


def create_conversation(title: str = "Untitled") -> int:
    """Create a new empty conversation and return its id."""
    with session_scope() as session:
        conv = Conversation(title=title)
        session.add(conv)
        session.flush()
        return conv.id


def get_or_create_conversation(conversation_id: int | None, title: str = "Untitled") -> int:
    """Return the given conversation id, or create a new one."""
    if conversation_id is None:
        return create_conversation(title=title)

    with session_scope() as session:
        conv = session.get(Conversation, conversation_id)
        if conv is None:
            conv = Conversation(id=conversation_id, title=title)
            session.add(conv)
            session.flush()
        return conv.id


def save_user_message(conversation_id: int, content: str) -> int:
    """Save a user message and return its id."""
    with session_scope() as session:
        msg = Message(conversation_id=conversation_id, role="user", content=content)
        session.add(msg)
        session.flush()
        return msg.id


def load_conversation_messages(conversation_id: int, limit: int = 50) -> list[dict[str, str]]:
    """Load the last N messages of a conversation, oldest first."""
    with session_scope() as session:
        rows = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()
        return [{"role": m.role, "content": m.content} for m in rows]


def save_agent_run(conversation_id: int, state: AgentState, model: str = "") -> int:
    """Save a completed agent run and all its tool calls."""
    with session_scope() as session:
        run = AgentRun(
            conversation_id=conversation_id,
            user_message=state.user_message,
            final_answer=state.final_answer,
            status=state.status,
            iterations=state.iteration,
            model=model,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.flush()

        for tc in state.tool_calls:
            result_preview = ""
            if tc.result is not None:
                try:
                    result_preview = json.dumps(tc.result, ensure_ascii=False)[:500]
                except (TypeError, ValueError):
                    result_preview = str(tc.result)[:500]

            session.add(
                ToolCall(
                    run_id=run.id,
                    iteration=tc.iteration,
                    tool_name=tc.name,
                    arguments_json=json.dumps(tc.arguments, ensure_ascii=False, default=str),
                    result_preview=result_preview,
                    error=tc.error,
                    duration_ms=0.0,
                )
            )

        # Also save the final assistant message to the conversation
        if state.final_answer:
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=state.final_answer,
                )
            )

        return run.id


def get_conversation_history(conversation_id: int) -> dict:
    """Return a summary of the conversation for inspection."""
    with session_scope() as session:
        conv = session.get(Conversation, conversation_id)
        if conv is None:
            return {}

        runs = (
            session.query(AgentRun)
            .filter(AgentRun.conversation_id == conversation_id)
            .order_by(AgentRun.id)
            .all()
        )

        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "runs": [
                {
                    "id": r.id,
                    "user_message": r.user_message,
                    "final_answer": r.final_answer,
                    "status": r.status,
                    "iterations": r.iterations,
                    "tool_calls": len(r.tool_calls),
                }
                for r in runs
            ],
        }