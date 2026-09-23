"""LangGraph agent endpoint with Human-in-the-Loop support.

Two endpoints:
  POST /graph/chat              → start a graph run (may pause for approval)
  POST /graph/approve/{thread}  → resume a paused run with a decision
"""

import json
import uuid

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from app.agent.graph.builder import build_graph
from app.agent.graph.checkpointer import get_checkpointer
from app.agent.llm import LLM, build_provider
from app.agent.observability import Tracer, silent_handler
from app.agent.rag.store import init_store
from app.agent.tools.registry import build_default_registry
from app.api.schemas import (
    ApprovalRequest,
    GraphChatRequest,
    GraphChatResponse,
    PendingApproval,
    ToolCallOut,
)
from app.core.config import settings
from app.db.base import init_db


router = APIRouter(tags=["graph"])


def _summarize_tool_calls(tool_calls: list[dict]) -> list[ToolCallOut]:
    out: list[ToolCallOut] = []
    for rec in tool_calls:
        try:
            preview = json.dumps(rec.get("result"), ensure_ascii=False)[:500]
        except (TypeError, ValueError):
            preview = str(rec.get("result"))[:500]

        out.append(
            ToolCallOut(
                iteration=rec.get("iteration", 0),
                tool_name=rec.get("name", "?"),
                arguments=rec.get("arguments", {}),
                result_preview=preview,
                error=rec.get("error"),
            )
        )
    return out


def _extract_pending(result: dict) -> PendingApproval | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    intr = interrupts[0]
    payload = intr.value if hasattr(intr, "value") else intr
    return PendingApproval(
        tool=payload.get("tool", "?"),
        arguments=payload.get("arguments", {}),
        iteration=payload.get("iteration", 0),
    )


def _build_initial_state(user_msg: str, system_prompt: str, conversation_id: int) -> dict:
    return {
        "user_message": user_msg,
        "conversation_id": conversation_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "iteration": 0,
        "max_iterations": 8,
        "recovery_count": 0,
        "tool_calls": [],
        "needs_tools": False,
        "final_answer": "",
        "status": "running",
    }


@router.post("/graph/chat", response_model=GraphChatResponse)
def graph_chat(request: GraphChatRequest) -> GraphChatResponse:
    """Start a LangGraph agent run. May pause for human approval."""
    from app.agent import memory
    from app.agent.loop import SYSTEM_PROMPT

    init_db()
    init_store()

    registry = build_default_registry()
    llm = LLM(provider=build_provider(settings.llm_provider))
    tracer = Tracer([silent_handler])

    # Persistence
    conversation_id = memory.get_or_create_conversation(
        request.conversation_id, title=request.message[:60]
    )
    memory.save_user_message(conversation_id, request.message)

    thread_id = f"conv-{conversation_id}-{uuid.uuid4().hex[:8]}"

    with get_checkpointer() as cp:
        graph = build_graph(registry, llm, tracer, checkpointer=cp)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = graph.invoke(
                _build_initial_state(request.message, SYSTEM_PROMPT, conversation_id),
                config=config,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph failed: {e}") from e

    pending = _extract_pending(result)

    if pending:
        return GraphChatResponse(
            status="paused",
            thread_id=thread_id,
            conversation_id=conversation_id,
            iterations=result.get("iteration", 0),
            pending_approval=pending,
            tool_calls=_summarize_tool_calls(result.get("tool_calls", [])),
        )

    # Done — save the run
    from app.agent.state import AgentState, ToolCallRecord

    legacy = AgentState(user_message=request.message)
    legacy.iteration = result.get("iteration", 0)
    legacy.status = result.get("status", "done")
    legacy.final_answer = result.get("final_answer", "")
    for rec in result.get("tool_calls", []):
        legacy.tool_calls.append(
            ToolCallRecord(
                name=rec["name"],
                arguments=rec["arguments"],
                result=rec.get("result"),
                error=rec.get("error"),
                iteration=rec.get("iteration", 0),
            )
        )
    memory.save_agent_run(conversation_id, legacy, model=llm.model)

    return GraphChatResponse(
        status=result.get("status", "done"),
        thread_id=thread_id,
        conversation_id=conversation_id,
        iterations=result.get("iteration", 0),
        final_answer=result.get("final_answer", ""),
        tool_calls=_summarize_tool_calls(result.get("tool_calls", [])),
    )


@router.post("/graph/approve/{thread_id}", response_model=GraphChatResponse)
def graph_approve(thread_id: str, request: ApprovalRequest) -> GraphChatResponse:
    """Resume a paused graph run with an approval decision."""
    registry = build_default_registry()
    llm = LLM(provider=build_provider(settings.llm_provider))
    tracer = Tracer([silent_handler])

    with get_checkpointer() as cp:
        graph = build_graph(registry, llm, tracer, checkpointer=cp)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = graph.invoke(
                Command(resume={"approved": request.approved}),
                config=config,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resume failed: {e}") from e

    # Possibly another interrupt (chained approvals)
    pending = _extract_pending(result)

    conversation_id = result.get("conversation_id", 0)

    if pending:
        return GraphChatResponse(
            status="paused",
            thread_id=thread_id,
            conversation_id=conversation_id,
            iterations=result.get("iteration", 0),
            pending_approval=pending,
            tool_calls=_summarize_tool_calls(result.get("tool_calls", [])),
        )

    # Save the completed run
    from app.agent import memory
    from app.agent.state import AgentState, ToolCallRecord

    legacy = AgentState(user_message=result.get("user_message", ""))
    legacy.iteration = result.get("iteration", 0)
    legacy.status = result.get("status", "done")
    legacy.final_answer = result.get("final_answer", "")
    for rec in result.get("tool_calls", []):
        legacy.tool_calls.append(
            ToolCallRecord(
                name=rec["name"],
                arguments=rec["arguments"],
                result=rec.get("result"),
                error=rec.get("error"),
                iteration=rec.get("iteration", 0),
            )
        )
    memory.save_agent_run(conversation_id, legacy, model=llm.model)

    return GraphChatResponse(
        status=result.get("status", "done"),
        thread_id=thread_id,
        conversation_id=conversation_id,
        iterations=result.get("iteration", 0),
        final_answer=result.get("final_answer", ""),
        tool_calls=_summarize_tool_calls(result.get("tool_calls", [])),
    )