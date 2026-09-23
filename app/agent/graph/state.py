"""GraphState — what flows through the LangGraph agent.

Design notes:
  - `messages` uses an "append" reducer: every node that returns
    {"messages": [...]} contributes to a growing conversation.
  - `tool_calls` similarly accumulates all tool invocations (for
    persistence + observability).
  - Everything else is replaced by whatever the node returns.
"""

from typing import Annotated, Any, TypedDict


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer: append new items to the existing list."""
    return (left or []) + (right or [])


class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph agent."""

    # --- Input ---
    user_message: str
    conversation_id: int | None

    # --- Conversation (full history as dicts: {role, content, tool_calls?}) ---
    messages: Annotated[list[dict[str, Any]], _append]

    # --- Loop control ---
    iteration: int
    max_iterations: int
    recovery_count: int

    # --- Tool activity (each item: {name, arguments, result, error, iteration}) ---
    tool_calls: Annotated[list[dict[str, Any]], _append]

    # --- Routing hint set by the LLM node ---
    needs_tools: bool

    # --- Output ---
    final_answer: str
    status: str