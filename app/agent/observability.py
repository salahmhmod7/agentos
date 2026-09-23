"""Lightweight observability layer.

Provides a simple event system the agent loop can emit to.
Events are printed in real time and (optionally) collected for later.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    ITERATION_START = "iteration_start"
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


EventHandler = Callable[[Event], None]


class Tracer:
    """Emit + collect agent events."""

    def __init__(self, handlers: list[EventHandler] | None = None):
        self.handlers: list[EventHandler] = handlers or []
        self.events: list[Event] = []

    def add_handler(self, handler: EventHandler) -> None:
        self.handlers.append(handler)

    def emit(self, event_type: EventType, **data: Any) -> None:
        event = Event(type=event_type, data=data)
        self.events.append(event)
        for handler in self.handlers:
            try:
                handler(event)
            except Exception:
                pass  # Never let a bad handler crash the agent


# ------------------------------------------------------------------
# Ready-made handlers
# ------------------------------------------------------------------

# ANSI colors (works on modern Windows Terminal / VS Code terminal)
_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
}


def _c(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def pretty_console_handler(event: Event) -> None:
    """Print a human-friendly line for each event."""
    t = event.type
    d = event.data

    if t == EventType.RUN_START:
        print(_c(f"\n▶ RUN START — {d.get('user_message', '')[:80]}", "bold"))
    elif t == EventType.RUN_END:
        dur = d.get("duration_s", 0)
        status = d.get("status", "?")
        color = "green" if status == "done" else "red"
        print(_c(f"■ RUN END — status={status} | total={dur:.1f}s", color))

    elif t == EventType.ITERATION_START:
        print(_c(f"\n  ⟳ iteration {d.get('iteration')}", "dim"))

    elif t == EventType.LLM_CALL_START:
        print(_c("    → LLM ...", "cyan"), end="", flush=True)
    elif t == EventType.LLM_CALL_END:
        dur = d.get("duration_s", 0)
        n_tools = d.get("tool_calls", 0)
        tag = f"{n_tools} tool call(s)" if n_tools else "final answer"
        print(_c(f" {dur:.1f}s ({tag})", "dim"))

    elif t == EventType.TOOL_CALL_START:
        args = d.get("arguments", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
        print(_c(f"    ⚙ {d.get('tool_name')}({args_str})", "yellow"), end="", flush=True)
    elif t == EventType.TOOL_CALL_END:
        dur = d.get("duration_s", 0)
        err = d.get("error")
        if err:
            print(_c(f" ✗ {dur:.2f}s — {err[:80]}", "red"))
        else:
            print(_c(f" ✓ {dur:.2f}s", "green"))

    elif t == EventType.ERROR:
        print(_c(f"    ! ERROR: {d.get('message', '')[:120]}", "red"))


def silent_handler(event: Event) -> None:
    """Do nothing — useful for tests."""
    pass
def make_sse_handler(queue):
    """Return a handler that pushes events into a queue (for SSE streaming)."""
    def handler(event: Event) -> None:
        queue.put(event)
    return handler