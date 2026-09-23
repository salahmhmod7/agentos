"""Assemble the LangGraph agent and provide a runnable entrypoint.

Graph shape:

    START → call_llm → (routing) → execute_tools → call_llm (loop)
                              ↘  finalize → END
                              ↘  force_end → END
"""

import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.graph.nodes import (
    finalize_node,
    force_end_node,
    make_call_llm_node,
    make_execute_tools_node,
)
from app.agent.graph.state import GraphState
from app.agent.llm import LLM, llm as default_llm
from app.agent.observability import EventType, Tracer, pretty_console_handler
from app.agent.state import AgentState, ToolCallRecord
from app.agent.tools.registry import ToolRegistry, build_default_registry


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_llm(state: dict) -> str:
    """Decide where to go after the LLM node."""
    status = state.get("status")
    if status in ("done", "failed"):
        return "finalize"

    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        return "force_end"

    if state.get("needs_tools"):
        return "execute_tools"

    return "finalize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(
    registry: ToolRegistry,
    llm: LLM,
    tracer: Tracer,
    checkpointer=None,
):
    """Create and compile the agent graph.

    If a checkpointer is provided, state persists after each node,
    enabling resume + human-in-the-loop.
    """
    builder = StateGraph(GraphState)

    builder.add_node("call_llm", make_call_llm_node(llm, registry, tracer))
    builder.add_node("execute_tools", make_execute_tools_node(registry, tracer))
    builder.add_node("finalize", finalize_node)
    builder.add_node("force_end", force_end_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {
            "execute_tools": "execute_tools",
            "finalize": "finalize",
            "force_end": "force_end",
        },
    )
    builder.add_edge("execute_tools", "call_llm")
    builder.add_edge("finalize", END)
    builder.add_edge("force_end", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_graph_agent(
    user_message: str,
    conversation_id: int | None = None,
    registry: ToolRegistry | None = None,
    llm: LLM | None = None,
    max_iterations: int = 8,
    tracer: Tracer | None = None,
    thread_id: str | None = None,
    checkpointer=None,
) -> tuple[AgentState, int, str]:
    """Run the agent using the LangGraph engine.

    Returns (AgentState, conversation_id, thread_id).
    """
    from app.agent import memory
    from app.agent.graph.checkpointer import get_checkpointer
    from app.agent.loop import SYSTEM_PROMPT

    registry = registry or build_default_registry()
    llm = llm or default_llm
    tracer = tracer or Tracer([pretty_console_handler])

    # --- Persistence: get/create conversation, save user message ---
    conversation_id = memory.get_or_create_conversation(
        conversation_id, title=user_message[:60]
    )
    memory.save_user_message(conversation_id, user_message)

    # --- Build the initial message history ---
    history = memory.load_conversation_messages(conversation_id, limit=6)

    initial_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    for msg in history[:-1]:
        initial_messages.append({"role": msg["role"], "content": msg["content"]})
    initial_messages.append({"role": "user", "content": user_message})

    # --- Thread id (used by the checkpointer) ---
    thread_id = thread_id or f"conv-{conversation_id}-{uuid.uuid4().hex[:8]}"

    # --- Tracer emit ---
    run_start = time.perf_counter()
    tracer.emit(
        EventType.RUN_START,
        user_message=user_message,
        model=llm.model,
        provider=llm.provider_name,
    )

    # --- Choose checkpointer (explicit or ephemeral in-memory) ---
    def _invoke(checkpointer_obj):
        graph = build_graph(registry, llm, tracer, checkpointer=checkpointer_obj)
        return graph.invoke(
            {
                "user_message": user_message,
                "conversation_id": conversation_id,
                "messages": initial_messages,
                "iteration": 0,
                "max_iterations": max_iterations,
                "recovery_count": 0,
                "tool_calls": [],
                "needs_tools": False,
                "final_answer": "",
                "status": "running",
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    if checkpointer is not None:
        final_state = _invoke(checkpointer)
    else:
        with get_checkpointer() as cp:
            final_state = _invoke(cp)

    # --- Convert to legacy AgentState for memory + callers ---
    legacy = AgentState(
        user_message=user_message,
        max_iterations=max_iterations,
    )
    legacy.iteration = final_state.get("iteration", 0)
    legacy.status = final_state.get("status", "done")
    legacy.final_answer = final_state.get("final_answer", "")

    for rec in final_state.get("tool_calls", []):
        legacy.tool_calls.append(
            ToolCallRecord(
                name=rec["name"],
                arguments=rec["arguments"],
                result=rec["result"],
                error=rec["error"],
                iteration=rec["iteration"],
            )
        )

    memory.save_agent_run(conversation_id, legacy, model=llm.model)

    tracer.emit(
        EventType.RUN_END,
        status=legacy.status,
        duration_s=time.perf_counter() - run_start,
        iterations=legacy.iteration,
    )

    return legacy, conversation_id, thread_id