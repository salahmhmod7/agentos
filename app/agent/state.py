"""Agent state — what flows through the agent loop.

This is a real, inspectable object. Every step of the loop
reads from it and writes to it.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """A single tool call made during an agent run."""

    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None = None
    iteration: int = 0


@dataclass
class AgentState:
    """Mutable state for one agent run."""

    # Input
    user_message: str = ""

    # Conversation (system + user + assistant + tool messages)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Tool activity
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # Loop control
    iteration: int = 0
    max_iterations: int = 10

    # Output
    final_answer: str | None = None
    status: str = "running"  # running | done | failed | max_iterations

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation."""
        self.messages.append({"role": role, "content": content})

    def add_tool_message(self, content: str) -> None:
        """Append a tool result message."""
        self.messages.append({"role": "tool", "content": content})