"""Tool registry — central place where all tools live."""

from typing import Any

from app.agent.tools.base import Tool
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.errors import ToolNotFoundError
from app.agent.tools.fetch_page import FetchPageTool
from app.agent.tools.find_in_page import FindInPageTool
from app.agent.tools.knowledge_search import KnowledgeSearchTool
from app.agent.tools.save_report import SaveReportTool
from app.agent.tools.web_search import WebSearchTool
from app.agent.tools.wikipedia import WikipediaTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for t in self._tools.values():
            s = t.to_ollama_schema()
            desc = s["function"]["description"]
            if len(desc) > 200:
                s["function"]["description"] = desc[:200] + "..."
            schemas.append(s)
        return schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        return tool.run(**arguments)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def requires_approval(self, name: str) -> bool:
        """Return True if the tool needs human approval before running."""
        try:
            return self.get(name).requires_approval
        except ToolNotFoundError:
            return False


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(KnowledgeSearchTool())
    registry.register(WikipediaTool())
    registry.register(WebSearchTool())
    registry.register(FetchPageTool())
    registry.register(FindInPageTool())
    registry.register(SaveReportTool())
    return registry