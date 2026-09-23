"""Document loader — turn PDF/TXT/MD files into plain text + metadata."""

from pathlib import Path
from typing import Any

import fitz


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        text, num_pages = _load_pdf(p)
    else:
        text = p.read_text(encoding="utf-8", errors="ignore")
        num_pages = 1

    return {
        "text": text,
        "metadata": {
            "filename": p.name,
            "extension": ext,
            "num_pages": num_pages,
        },
    }


def _load_pdf(path: Path) -> tuple[str, int]:
    """Extract text from a PDF using PyMuPDF."""
    parts: list[str] = []
    with fitz.open(path) as doc:
        # IMPORTANT: capture page_count INSIDE the with-block
        page_count = doc.page_count
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts), page_count