"""Full ingest pipeline: load → clean → chunk → embed → store."""

from pathlib import Path
from typing import Any

from app.agent.rag.chunker import chunk_text
from app.agent.rag.cleaner import clean_text
from app.agent.rag.embedder import embedder
from app.agent.rag.loader import load_document
from app.agent.rag.store import init_store, insert_document


def ingest_file(
    path: str | Path,
    chunk_size: int = 800,
    overlap: int = 120,
) -> dict[str, Any]:
    init_store()

    p = Path(path)
    doc = load_document(p)
    raw_chars = len(doc["text"])

    cleaned = clean_text(doc["text"])
    cleaned_chars = len(cleaned)

    chunks = chunk_text(
        cleaned,
        chunk_size=chunk_size,
        overlap=overlap,
        base_metadata={"source": doc["metadata"]["filename"]},
    )

    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    document_id = insert_document(
        name=doc["metadata"]["filename"],
        extension=doc["metadata"]["extension"],
        num_pages=doc["metadata"]["num_pages"],
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "document_id": document_id,
        "filename": doc["metadata"]["filename"],
        "num_pages": doc["metadata"]["num_pages"],
        "raw_chars": raw_chars,
        "cleaned_chars": cleaned_chars,
        "num_chunks": len(chunks),
    }