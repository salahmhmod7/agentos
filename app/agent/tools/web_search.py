"""Web search tool — uses DuckDuckGo (no API key needed).

DuckDuckGo is free but flaky: it rate-limits automated requests and
sometimes returns totally unrelated results. We retry with backoff,
and we return a structured payload with an explicit note telling the
LLM not to hallucinate.
"""

import time
from typing import Any

from ddgs import DDGS

from app.agent.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web and return the top results."""

    name = "web_search"
    description = (
        "Search the web for current information. "
        "Returns a list of results with title, URL, and a short snippet. "
        "IMPORTANT: the 'note' field tells you how to interpret the results. "
        "If count is 0, nothing was found — do NOT invent an answer."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return (1-10). Default 5.",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5, **_: Any) -> dict[str, Any]:
        """Execute a search with retries. Never raises on empty results."""
        max_results = max(1, min(int(max_results), 10))

        last_error: str | None = None

        # Try up to 3 times with backoff
        for attempt in range(3):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * (attempt + 1))
                continue

            if raw:
                results = [
                    {
                        "title": item.get("title", "").strip(),
                        "url": item.get("href", "").strip(),
                        "snippet": item.get("body", "").strip(),
                    }
                    for item in raw
                ]
                return {
                    "query": query,
                    "count": len(results),
                    "results": results,
                    "note": (
                        "These are the ONLY facts available. Do NOT state any "
                        "number/date/version unless it appears literally in a "
                        "snippet above. If none of these snippets answers the "
                        "question, say so honestly."
                    ),
                }

            # Empty result — wait and retry
            time.sleep(1.5 * (attempt + 1))

        # All attempts done. Return structured "nothing found" signal.
        return {
            "query": query,
            "count": 0,
            "results": [],
            "note": (
                "No results found after retries. This is likely a DuckDuckGo "
                "rate limit or a query with no matches. Do NOT invent an answer. "
                "Try a different query, or use fetch_page on a known URL."
            ),
            "last_error": last_error,
        }