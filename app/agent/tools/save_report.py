"""Save-report tool — writes a markdown file to disk.

This is a WRITE operation, so it requires human approval.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.tools.base import Tool
from app.agent.tools.errors import ToolExecutionError


REPORTS_DIR = Path("data/reports")


class SaveReportTool(Tool):
    """Save a markdown report to data/reports/."""

    name = "save_report"
    description = (
        "Save a markdown report to disk under data/reports/. "
        "Use this when the user explicitly asks you to save the answer as a file. "
        "This is a WRITE operation and requires human approval before execution. "
        "Provide a short filename (no path) and the full report content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename like 'ada_lovelace_report.md'. Must end in .md.",
            },
            "content": {
                "type": "string",
                "description": "The full markdown content of the report.",
            },
        },
        "required": ["filename", "content"],
    }

    requires_approval = True  # <-- triggers HITL

    def run(self, filename: str, content: str, **_: Any) -> dict[str, Any]:
        # Sanitize filename
        name = Path(filename).name
        if not name.endswith(".md"):
            name = name + ".md"
        if not name or name == ".md":
            raise ToolExecutionError("A valid filename is required.")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / name

        try:
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise ToolExecutionError(f"Failed to write {path}: {e}") from e

        return {
            "status": "saved",
            "path": str(path),
            "bytes": len(content.encode("utf-8")),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }