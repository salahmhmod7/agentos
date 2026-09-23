"""Chat endpoint — the main entry point to the agent."""

import json

from fastapi import APIRouter, HTTPException

from app.agent.loop import run_agent
from app.agent.observability import EventType, Tracer, silent_handler
from app.api.schemas import ChatRequest, ChatResponse, ToolCallOut


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the agent synchronously and return the final answer."""
    try:
        tracer = Tracer([silent_handler])
        state, conversation_id = run_agent(
            user_message=request.message,
            conversation_id=request.conversation_id,
            tracer=tracer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}") from e

    # Build the response
    tools_out: list[ToolCallOut] = []
    for tc in state.tool_calls:
        try:
            preview = json.dumps(tc.result, ensure_ascii=False)[:500]
        except (TypeError, ValueError):
            preview = str(tc.result)[:500]

        tools_out.append(
            ToolCallOut(
                iteration=tc.iteration,
                tool_name=tc.name,
                arguments=tc.arguments,
                result_preview=preview,
                error=tc.error,
            )
        )

    # Fetch run_id from the last run of this conversation
    from app.agent import memory

    history = memory.get_conversation_history(conversation_id)
    run_id = history["runs"][-1]["id"] if history.get("runs") else None

    return ChatResponse(
        conversation_id=conversation_id,
        run_id=run_id,
        status=state.status,
        iterations=state.iteration,
        final_answer=state.final_answer,
        tool_calls=tools_out,
    )