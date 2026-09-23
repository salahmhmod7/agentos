"""Find-in-page tool — search for a pattern inside a text block."""

import re
from typing import Any

from app.agent.tools.base import Tool
from app.agent.tools.errors import ToolExecutionError


class FindInPageTool(Tool):
    """Search inside a block of text and return matches with context."""

    name = "find_in_page"
    description = (
        "Search for a literal text pattern inside a body of text you already have. "
        "Returns each match with surrounding context (default 200 chars). "
        "Use this when you fetched a page or a document excerpt and need to locate "
        "a specific keyword/phrase inside it. Case-insensitive."
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The text to search for (literal, not regex).",
            },
            "text": {
                "type": "string",
                "description": "The text to search inside. Pass the whole page/excerpt.",
            },
            "context_chars": {
                "type": "integer",
                "description": "Chars of context around each match. Default 200.",
            },
            "max_matches": {
                "type": "integer",
                "description": "Max matches to return. Default 10.",
            },
        },
        "required": ["pattern", "text"],
    }

    def run(
        self,
        pattern: str,
        text: str,
        context_chars: int = 200,
        max_matches: int = 10,
        **_: Any,
    ) -> dict[str, Any]:
        if not pattern or not text:
            raise ToolExecutionError("Both 'pattern' and 'text' are required.")

        context_chars = max(20, min(int(context_chars), 2000))
        max_matches = max(1, min(int(max_matches), 50))

        try:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        except re.error as e:
            raise ToolExecutionError(f"Invalid pattern: {e}") from e

        matches = []
        for m in regex.finditer(text):
            start = max(0, m.start() - context_chars)
            end = min(len(text), m.end() + context_chars)
            matches.append(
                {
                    "position": m.start(),
                    "context": text[start:end],
                }
            )
            if len(matches) >= max_matches:
                break

        return {
            "pattern": pattern,
            "text_length": len(text),
            "match_count": len(matches),
            "matches": matches,
            "note": (
                "If match_count is 0, the pattern was NOT found in the text. "
                "Do not invent content — say it's not present."
            ),
        }