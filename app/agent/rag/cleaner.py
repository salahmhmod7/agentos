"""Text cleaner — normalize raw document text."""

import re


_LINE_NOISE_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*-{3,}\s*$"),
    re.compile(r"^\s*=+\s*$"),
]

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    kept = [line for line in lines if not _is_noise(line)]
    text = "\n".join(kept)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _LINE_NOISE_PATTERNS)
