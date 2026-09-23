"""Base Tool abstraction.

Every tool in AgentOS exposes:
- name: unique identifier the LLM uses to call it
- description: natural-language explanation for the LLM
- parameters: JSON schema describing the arguments
- requires_approval: if True, the graph pauses for human approval
- run(): actual Python function
"""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all AgentOS tools."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False  # <-- new

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }