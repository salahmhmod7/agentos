"""Document upload + management endpoints."""

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent.rag.ingest import ingest_file
from app.agent.rag.store import DB_PATH, delete_document, list_documents
from app.api.schemas import ChunkPreview, DocumentDetail, DocumentInfo


router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/documents", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Upload a document, ingest it into the knowledge base."""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {MAX_FILE_SIZE // (1024*1024)} MB.",
            )
        tmp.write(content)
        tmp_path = Path(tmp.name)

    final_path = tmp_path.parent / filename
    shutil.move(str(tmp_path), str(final_path))

    try:
        summary = ingest_file(final_path)
    except Exception as e:
        final_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e
    finally:
        final_path.unlink(missing_ok=True)

    return DocumentInfo(**summary)


@router.get("/documents", response_model=list[DocumentInfo])
def get_documents() -> list[DocumentInfo]:
    """List all documents in the knowledge base."""
    return [DocumentInfo(**d) for d in list_documents()]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int) -> DocumentDetail:
    """Return a document and its chunks (for preview)."""
    docs = [d for d in list_documents() if d["id"] == document_id]
    if not docs:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = docs[0]

    # Read chunks directly (no vector search — just fetch text)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, chunk_index, text, metadata_json
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (document_id,),
        ).fetchall()
    finally:
        conn.close()

    chunks = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["metadata_json"] or "{}")
        except json.JSONDecodeError:
            pass
        chunks.append(
            ChunkPreview(
                id=r["id"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                metadata=meta,
            )
        )

    return DocumentDetail(**doc, chunks=chunks)


@router.delete("/documents/{document_id}")
def remove_document(document_id: int) -> dict:
    """Delete a document and its chunks/embeddings."""
    ok = delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}
