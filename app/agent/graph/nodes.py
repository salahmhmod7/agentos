"""Node functions for the LangGraph agent.

Each node:
  - reads the current GraphState
  - does ONE thing (call LLM / run tools / finalize)
  - returns a dict of *updates* to the state
"""

import json
import re
import time
from typing import Any, Callable

from app.agent.llm import LLM
from app.agent.observability import EventType, Tracer
from app.agent.tools.registry import ToolRegistry
from langgraph.types import interrupt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_tool_result(result: Any, max_chars: int = 2500) -> str:
    if isinstance(result, (dict, list)):
        s = json.dumps(result, ensure_ascii=False)
    else:
        s = str(result)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n...[truncated, {len(s) - max_chars} chars omitted]"
    return s


def _find_last_assistant_with_tools(messages: list[dict]) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return m
    return None


# ---------------------------------------------------------------------------
# Node factories (closure over llm / registry / tracer)
# ---------------------------------------------------------------------------

def make_call_llm_node(
    llm: LLM,
    registry: ToolRegistry,
    tracer: Tracer,
) -> Callable[[dict], dict]:
    """Node: send the current messages to the LLM and store its response."""
    tool_schemas = registry.schemas()

    def call_llm(state: dict) -> dict:
        iteration = state.get("iteration", 0) + 1
        tracer.emit(EventType.ITERATION_START, iteration=iteration)

        messages = list(state.get("messages", []))

        # Force answer if we're near the iteration cap
        max_iters = state.get("max_iterations", 8)
        if iteration >= max_iters - 1:
            messages.append({
                "role": "system",
                "content": (
                    "You are near the iteration limit. Answer NOW with the "
                    "information you have. Do NOT call any more tools."
                ),
            })

        # --- LLM call ---
        tracer.emit(EventType.LLM_CALL_START)
        t0 = time.perf_counter()

        try:
            response = llm.chat_with_tools(messages=messages, tools=tool_schemas)
        except Exception as e:
            err_str = str(e)

            # Special case: invalid tool name (Groq rejects the whole request)
            if "tool_use_failed" in err_str or "not in request.tools" in err_str:
                m = re.search(r"tool '([^']+)'", err_str)
                bad_tool = m.group(1) if m else "an unknown tool"
                recovery = state.get("recovery_count", 0) + 1

                tracer.emit(
                    EventType.ERROR,
                    message=f"Invalid tool '{bad_tool}' — injecting recovery hint",
                )

                if recovery > 3:
                    return {
                        "status": "failed",
                        "final_answer": (
                            f"Agent kept calling invalid tools after 3 attempts. "
                            f"Last invalid tool: {bad_tool}"
                        ),
                        "needs_tools": False,
                    }

                return {
                    "messages": [{
                        "role": "system",
                        "content": (
                            f"STOP. You tried to call '{bad_tool}', which is NOT "
                            f"a valid tool. The ONLY valid tools are: "
                            f"{', '.join(registry.names())}. "
                            f"Continue using one of those."
                        ),
                    }],
                    "recovery_count": recovery,
                    "iteration": iteration - 1,  # don't count this as an iteration
                    "needs_tools": True,
                }

            # Regular LLM failure
            tracer.emit(EventType.ERROR, message=err_str)
            return {
                "status": "failed",
                "final_answer": f"LLM call failed: {e}",
                "needs_tools": False,
            }

        llm_dur = time.perf_counter() - t0
        tracer.emit(
            EventType.LLM_CALL_END,
            duration_s=llm_dur,
            tool_calls=len(response.tool_calls),
        )

        needs_tools = bool(response.tool_calls)

        # Build the assistant message (with tool_calls if any)
        assistant_msg = dict(response.raw_message)
        if not assistant_msg.get("content") and response.content:
            assistant_msg["content"] = response.content

        updates: dict = {
            "messages": [assistant_msg],
            "iteration": iteration,
            "needs_tools": needs_tools,
        }

        # If no tools needed → this is the final answer
        if not needs_tools:
            updates["status"] = "done"
            updates["final_answer"] = response.content or ""

        return updates

    return call_llm

def make_execute_tools_node(
    registry: ToolRegistry,
    tracer: Tracer,
) -> Callable[[dict], dict]:
    """Node: execute every tool call in the last assistant message.

    If a tool requires approval, we `interrupt()` — LangGraph pauses here
    and the caller can resume with a decision via Command(resume=...).
    """

    def execute_tools(state: dict) -> dict:
        messages = list(state.get("messages", []))
        last_assistant = _find_last_assistant_with_tools(messages)
        if not last_assistant:
            return {"needs_tools": False}

        iteration = state.get("iteration", 1)
        new_messages: list[dict] = []
        new_records: list[dict] = []

        for tc in last_assistant.get("tool_calls", []):
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            else:
                args = raw_args or {}

            # --- HITL: pause if this tool requires approval ---
            if registry.requires_approval(name):
                # This call raises a special exception that LangGraph catches.
                # The caller must resume with Command(resume={"approved": bool}).
                decision = interrupt(
                    {
                        "action": "approval_required",
                        "tool": name,
                        "arguments": args,
                        "iteration": iteration,
                    }
                )
                # On resume, `decision` is whatever the caller passed.
                approved = bool(decision.get("approved")) if isinstance(decision, dict) else False

                if not approved:
                    new_records.append({
                        "name": name,
                        "arguments": args,
                        "result": None,
                        "error": "REJECTED by human reviewer",
                        "iteration": iteration,
                    })
                    new_messages.append({
                        "role": "tool",
                        "content": (
                            "ERROR: The human reviewer rejected this action. "
                            "Do not retry it. Find another way to answer, or "
                            "explain to the user that the action was rejected."
                        ),
                        **({"tool_call_id": tc["id"]} if tc.get("id") else {}),
                        **({"name": name} if tc.get("id") else {}),
                    })
                    continue

            # --- Execute the tool ---
            tracer.emit(
                EventType.TOOL_CALL_START,
                tool_name=name,
                arguments=args,
            )

            t0 = time.perf_counter()
            result = None
            error = None
            try:
                result = registry.execute(name, args)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            tool_dur = time.perf_counter() - t0

            tracer.emit(
                EventType.TOOL_CALL_END,
                tool_name=name,
                duration_s=tool_dur,
                error=error,
            )

            new_records.append({
                "name": name,
                "arguments": args,
                "result": result,
                "error": error,
                "iteration": iteration,
            })

            content = (
                _format_tool_result(result)
                if error is None
                else f"ERROR: {error}"
            )
            tool_msg: dict[str, Any] = {"role": "tool", "content": content}
            if tc.get("id"):
                tool_msg["tool_call_id"] = tc["id"]
                tool_msg["name"] = name
            new_messages.append(tool_msg)

        return {
            "messages": new_messages,
            "tool_calls": new_records,
            "needs_tools": False,
        }

    return execute_tools


def finalize_node(state: dict) -> dict:
    """Node: wrap up if the LLM didn't already produce a final answer.

    LangGraph requires every node to write at least one key, so we
    always return the `status` field — even if nothing else changed.
    """
    # Already have a final answer from call_llm — just make sure status is set
    if state.get("final_answer"):
        return {"status": state.get("status") or "done"}

    # Fall back to the last assistant message
    last_content = ""
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "assistant":
            last_content = m.get("content") or ""
            break

    return {
        "final_answer": last_content or "(no answer produced)",
        "status": state.get("status") or "done",
    }

def force_end_node(state: dict) -> dict:
    """Node: hit the iteration cap without a clean answer."""
    return {
        "status": "max_iterations",
        "final_answer": "(Agent hit max_iterations without finishing.)",
        "needs_tools": False,
    }