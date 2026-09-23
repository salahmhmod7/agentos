"""Fetch a web page and return its readable text."""

from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.agent.tools.base import Tool
from app.agent.tools.errors import ToolExecutionError


# Tags to throw away entirely
_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript"]


class FetchPageTool(Tool):
    """Fetch a URL and extract the main readable text."""

    name = "fetch_page"
    description = (
        "Fetch the content of a web page and return its main text. "
        "Use this after `web_search` when you need the full content of a result. "
        "Input must be a full URL starting with http:// or https://."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of text to return. Default 4000.",
            },
        },
        "required": ["url"],
    }

    def run(self, url: str, max_chars: int = 4000, **_: Any) -> dict[str, str]:
        """Fetch, clean, and return the page's text."""
        if not url.startswith(("http://", "https://")):
            raise ToolExecutionError(f"Invalid URL (must start with http): {url}")

        max_chars = max(500, min(int(max_chars), 20000))

        try:
            with httpx.Client(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "AgentOS/0.1 (+research bot)"},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPError as e:
            raise ToolExecutionError(f"HTTP error fetching {url}: {e}") from e

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        return {
            "url": url,
            "title": (soup.title.string.strip() if soup.title and soup.title.string else ""),
            "text": cleaned[:max_chars],
        }