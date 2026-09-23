"""The agent loop."""

import json
import re
import time
from typing import Any

from app.agent.llm import LLM, llm as default_llm
from app.agent.observability import EventType, Tracer, pretty_console_handler
from app.agent.state import AgentState, ToolCallRecord
from app.agent.tools.registry import ToolRegistry, build_default_registry


SYSTEM_PROMPT = """You are AgentOS, a careful research assistant.

TOOLS (use the right one):
- knowledge_search: user's uploaded docs
- wikipedia: established public facts
- calculator: arithmetic
- web_search: current events/prices
- fetch_page: read a URL
- find_in_page: search inside text you have

RULES (strict):
1. Every fact must come from a tool call in THIS turn.
2. NEVER invent tool names. Only use tools from the schema.
3. If a tool returns nothing, say "Not found in the book/docs".
4. Be concise. Max 3 sentences unless the user asks for detail.

CITATION (mandatory when using knowledge_search):
- End every factual sentence with a citation in this exact format:
  [<filename>, chunk <chunk_index>]
- If you use multiple chunks, cite each one.
- Example: "Employees receive 21 days of annual leave [handbook.pdf, chunk 0]."
- If you cannot cite a claim from a tool result, do not make the claim.
"""


def _call_llm_with_manual_retry(
    llm: LLM,
    messages: list,
    tools: list,
    max_attempts: int = 3,
):
    """Call the LLM with manual retry on transient errors.

    IMPORTANT: never retries on 'tool_use_failed' — that error needs
    corrective messages from the caller, not blind retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return llm.chat_with_tools(messages=messages, tools=tools)
        except Exception as e:
            err_str = str(e)
            # Invalid tool call — do not retry, let caller recover
            if "tool_use_failed" in err_str or "not in request.tools" in err_str:
                raise
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _format_tool_result(result: Any, max_chars: int = 2500) -> str:
    """Convert tool result to string, truncating to keep context small."""
    if isinstance(result, (dict, list)):
        s = json.dumps(result, ensure_ascii=False)
    else:
        s = str(result)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n...[truncated, {len(s) - max_chars} chars omitted]"
    return s


def run_agent(
    user_message: str,
    conversation_id: int | None = None,
    registry: ToolRegistry | None = None,
    llm: LLM | None = None,
    max_iterations: int = 8,
    tracer: Tracer | None = None,
) -> tuple[AgentState, int]:
    from app.agent import memory

    registry = registry or build_default_registry()
    llm = llm or default_llm
    tracer = tracer or Tracer([pretty_console_handler])

    run_start = time.perf_counter()
    tracer.emit(
        EventType.RUN_START,
        user_message=user_message,
        model=llm.model,
        provider=llm.provider_name,
    )

    conversation_id = memory.get_or_create_conversation(
        conversation_id,
        title=user_message[:60],
    )
    memory.save_user_message(conversation_id, user_message)

    # Keep history SHORT to avoid 413 errors from Groq
    history = memory.load_conversation_messages(conversation_id, limit=6)

    state = AgentState(user_message=user_message, max_iterations=max_iterations)
    state.add_message("system", SYSTEM_PROMPT)
    for msg in history[:-1]:
        state.add_message(msg["role"], msg["content"])
    state.add_message("user", user_message)

    tool_schemas = registry.schemas()
    recovery_count = 0
    max_recoveries = 3

    while state.iteration < state.max_iterations:
        state.iteration += 1
        tracer.emit(EventType.ITERATION_START, iteration=state.iteration)

        # Force answer if we're near the iteration cap
        if state.iteration >= state.max_iterations - 1:
            state.messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are near the iteration limit. "
                        "Answer NOW with the information you have. "
                        "Do NOT call any more tools."
                    ),
                }
            )

        tracer.emit(EventType.LLM_CALL_START)
        t0 = time.perf_counter()

        try:
            response = _call_llm_with_manual_retry(llm, state.messages, tool_schemas)
        except Exception as e:
            err_str = str(e)

            # Recovery path: LLM invented an invalid tool name.
            # Groq rejects the entire request with 400 tool_use_failed.
            if "tool_use_failed" in err_str or "not in request.tools" in err_str:
                m = re.search(r"tool '([^']+)'", err_str)
                bad_tool = m.group(1) if m else "an unknown tool"

                tracer.emit(
                    EventType.ERROR,
                    message=f"Invalid tool '{bad_tool}' — injecting recovery hint",
                )

                recovery_count += 1
                if recovery_count > max_recoveries:
                    state.status = "failed"
                    state.final_answer = (
                        f"Agent kept calling invalid tools after {max_recoveries} "
                        f"attempts. Last invalid tool: {bad_tool}"
                    )
                    memory.save_agent_run(conversation_id, state, model=llm.model)
                    tracer.emit(
                        EventType.RUN_END,
                        status=state.status,
                        duration_s=time.perf_counter() - run_start,
                    )
                    return state, conversation_id

                # Inject corrective message and retry without advancing iteration
                state.messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"STOP. You just tried to call '{bad_tool}', which is NOT "
                            f"a valid tool and will be rejected by the API. "
                            f"The ONLY valid tools are: {', '.join(registry.names())}. "
                            f"Do NOT invent tool names. Continue the task now using "
                            f"one of the valid tools."
                        ),
                    }
                )
                state.iteration -= 1
                continue

            # Regular failure (includes 413 too-large and 429 rate-limit)
            tracer.emit(EventType.ERROR, message=err_str)
            state.status = "failed"
            state.final_answer = f"LLM call failed after retries: {e}"
            memory.save_agent_run(conversation_id, state, model=llm.model)
            tracer.emit(
                EventType.RUN_END,
                status=state.status,
                duration_s=time.perf_counter() - run_start,
            )
            return state, conversation_id

        llm_dur = time.perf_counter() - t0
        tracer.emit(
            EventType.LLM_CALL_END,
            duration_s=llm_dur,
            tool_calls=len(response.tool_calls),
        )

        # Case 1: tool calls
        if response.tool_calls:
            state.messages.append(response.raw_message)

            for call in response.tool_calls:
                tracer.emit(
                    EventType.TOOL_CALL_START,
                    tool_name=call.name,
                    arguments=call.arguments,
                )

                record = ToolCallRecord(
                    name=call.name,
                    arguments=call.arguments,
                    result=None,
                    iteration=state.iteration,
                )

                t0 = time.perf_counter()
                try:
                    record.result = registry.execute(call.name, call.arguments)
                except Exception as e:
                    record.error = f"{type(e).__name__}: {e}"
                tool_dur = time.perf_counter() - t0

                tracer.emit(
                    EventType.TOOL_CALL_END,
                    tool_name=call.name,
                    duration_s=tool_dur,
                    error=record.error,
                )

                state.tool_calls.append(record)

                content = (
                    _format_tool_result(record.result)
                    if record.error is None
                    else f"ERROR: {record.error}"
                )

                tool_msg: dict[str, Any] = {"role": "tool", "content": content}
                if call.id:
                    tool_msg["tool_call_id"] = call.id
                    tool_msg["name"] = call.name
                state.messages.append(tool_msg)

            continue

        # Case 2: final answer
        state.final_answer = response.content
        state.status = "done"
        state.add_message("assistant", response.content)
        memory.save_agent_run(conversation_id, state, model=llm.model)
        tracer.emit(
            EventType.RUN_END,
            status=state.status,
            duration_s=time.perf_counter() - run_start,
            iterations=state.iteration,
        )
        return state, conversation_id

    state.status = "max_iterations"
    state.final_answer = "(Agent hit max_iterations without finishing.)"
    memory.save_agent_run(conversation_id, state, model=llm.model)
    tracer.emit(
        EventType.RUN_END,
        status=state.status,
        duration_s=time.perf_counter() - run_start,
    )
    return state, conversation_id