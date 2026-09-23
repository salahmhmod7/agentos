"""Wikipedia tool — reliable, free, no API key needed.

Uses Wikipedia's public REST API directly via httpx.
Two operations:
  - search: find articles matching a query
  - summary: fetch the summary of a specific article

This tool is designed to be trustworthy: it returns EXACT text
from Wikipedia so the LLM can quote it verbatim without hallucinating.
"""

from typing import Any
from urllib.parse import quote

import httpx

from app.agent.tools.base import Tool
from app.agent.tools.errors import ToolExecutionError


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"
USER_AGENT = "AgentOS/0.1 (research bot; contact: local)"


class WikipediaTool(Tool):
    """Search Wikipedia and fetch article summaries."""

    name = "wikipedia"
    description = (
        "Search Wikipedia or get a summary of a Wikipedia article. "
        "Use this for factual, well-established knowledge: people, places, "
        "historical events, scientific concepts, technologies. "
        "Set 'action' to 'search' to find articles, or 'summary' to read one. "
        "Wikipedia text is reliable — quote it verbatim when you use it."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "summary"],
                "description": (
                    "'search' finds articles matching a query. "
                    "'summary' fetches the intro of a specific article title."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "For 'search': the search query. "
                    "For 'summary': the exact article title (e.g. 'Python (programming language)')."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "For 'search': how many results (1-10). Default 5.",
            },
        },
        "required": ["action", "query"],
    }

    def run(self, action: str, query: str, max_results: int = 5, **_: Any) -> dict[str, Any]:
        if action == "search":
            return self._search(query, max_results)
        elif action == "summary":
            return self._summary(query)
        else:
            raise ToolExecutionError(f"Unknown action: {action!r}. Use 'search' or 'summary'.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> dict[str, Any]:
        max_results = max(1, min(int(max_results), 10))

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }

        try:
            with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
                response = client.get(WIKI_API, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"Wikipedia search failed: {e}") from e

        hits = data.get("query", {}).get("search", [])

        results = [
            {
                "title": hit.get("title", ""),
                "snippet": _strip_html(hit.get("snippet", "")),
                "url": f"https://en.wikipedia.org/wiki/{quote(hit.get('title', '').replace(' ', '_'))}",
            }
            for hit in hits
        ]

        return {
            "action": "search",
            "query": query,
            "count": len(results),
            "results": results,
            "note": (
                "Wikipedia search results. To read an article, call "
                "wikipedia(action='summary', query='<exact title>')."
            ),
        }

    def _summary(self, title: str) -> dict[str, Any]:
        url = f"{WIKI_SUMMARY}/{quote(title.replace(' ', '_'))}"

        try:
            with httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT}) as client:
                response = client.get(url)
                if response.status_code == 404:
                    return {
                        "action": "summary",
                        "query": title,
                        "found": False,
                        "note": (
                            f"No Wikipedia article found with title '{title}'. "
                            "Try wikipedia(action='search', query='...') first to "
                            "find the exact title."
                        ),
                    }
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"Wikipedia summary failed: {e}") from e

        extract = data.get("extract", "").strip()

        return {
            "action": "summary",
            "query": title,
            "found": True,
            "title": data.get("title", title),
            "extract": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "note": (
                "This is the official Wikipedia intro. Quote it verbatim "
                "when you cite facts from it."
            ),
        }


def _strip_html(text: str) -> str:
    """Remove HTML tags from Wikipedia snippets."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()