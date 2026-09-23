"""Chunker — split a document into overlapping chunks."""

from typing import Any


DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    base_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_metadata = base_metadata or {}

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    pieces = _recursive_split(text, separators, chunk_size)
    chunks = _merge_with_overlap(pieces, chunk_size, overlap)

    out: list[dict[str, Any]] = []
    for i, chunk_str in enumerate(chunks):
        chunk_str = chunk_str.strip()
        if not chunk_str:
            continue
        out.append(
            {
                "text": chunk_str,
                "chunk_index": i,
                "metadata": {**base_metadata, "chunk_index": i},
            }
        )
    return out


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    parts = [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]

    out: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            out.append(part)
        else:
            if not rest:
                out.append(part)
            else:
                out.extend(_recursive_split(part, rest, chunk_size))
    return out


def _merge_with_overlap(
    pieces: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    merged: list[str] = []
    buf = ""

    for piece in pieces:
        if not piece:
            continue
        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            if buf:
                merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)

    if overlap <= 0 or len(merged) <= 1:
        return merged

    out: list[str] = [merged[0]]
    for i in range(1, len(merged)):
        tail = out[-1][-overlap:]
        out.append(tail + merged[i])
    return out