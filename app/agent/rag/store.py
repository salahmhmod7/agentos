"""Vector store backed by sqlite-vec."""

import json
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import sqlite_vec


def _resolve_db_path() -> str:
    url = "sqlite:///./agentos.db"
    try:
        from app.core.config import settings
        url = settings.database_url
    except Exception:
        pass
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "", 1)
    return url


DB_PATH = _resolve_db_path()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_store() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                extension TEXT,
                num_pages INTEGER DEFAULT 1,
                num_chunks INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (document_id) REFERENCES rag_documents(id)
                    ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);
            """
        )


def insert_document(
    name: str,
    extension: str,
    num_pages: int,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
        )

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO rag_documents (name, extension, num_pages, num_chunks, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                extension,
                num_pages,
                len(chunks),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        document_id = cur.lastrowid

        for chunk, emb in zip(chunks, embeddings):
            cur = conn.execute(
                """
                INSERT INTO document_chunks
                    (document_id, chunk_index, text, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk["chunk_index"],
                    chunk["text"],
                    json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                ),
            )
            chunk_id = cur.lastrowid

            conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, _to_blob(emb)),
            )

        return document_id


def search(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    with _connect() as conn:
        sql = """
            SELECT
                v.chunk_id AS chunk_id,
                v.distance AS distance,
                c.text AS text,
                c.chunk_index AS chunk_index,
                c.metadata_json AS metadata_json,
                d.id AS document_id,
                d.name AS document_name
            FROM vec_chunks v
            JOIN document_chunks c ON c.id = v.chunk_id
            JOIN rag_documents d ON d.id = c.document_id
            WHERE v.embedding MATCH ?
              AND k = ?
        """
        params: list[Any] = [_to_blob(query_embedding), top_k]

        if document_id is not None:
            sql = sql.replace("AND k = ?", "AND k = ? AND d.id = ?")
            params.append(document_id)

        sql += " ORDER BY v.distance"

        rows = conn.execute(sql, params).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            meta = {}
            try:
                meta = json.loads(r["metadata_json"] or "{}")
            except json.JSONDecodeError:
                pass
            out.append(
                {
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "chunk_index": r["chunk_index"],
                    "text": r["text"],
                    "distance": float(r["distance"]),
                    "metadata": meta,
                }
            )
        return out


def list_documents() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, extension, num_pages, num_chunks, created_at
            FROM rag_documents
            ORDER BY id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def delete_document(document_id: int) -> bool:
    with _connect() as conn:
        chunk_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        ]
        if not chunk_ids:
            return False

        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            (document_id,),
        )
        conn.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))
        return True


def _to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)