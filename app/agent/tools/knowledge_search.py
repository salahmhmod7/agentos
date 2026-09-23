"""Knowledge base search tool — RAG retrieval.

This tool lets the agent search the user's uploaded documents
(PDF, TXT, MD) using semantic similarity.
"""

from typing import Any

from app.agent.tools.base import Tool
from app.agent.tools.errors import ToolExecutionError
from app.agent.rag.embedder import embedder
from app.agent.rag.store import search


class KnowledgeSearchTool(Tool):
    """Search the user's uploaded documents."""

    name = "knowledge_search"
    description = (
        "Search the user's personal knowledge base (uploaded documents: "
        "PDFs, text files, markdown). "
        "Use this when the question is about the user's own content. "
        "Returns the top matching excerpts with their source filenames."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — a natural-language question or phrase.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many chunks to return (1-5). Default 3.",
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, top_k: int = 3, **_: Any) -> dict[str, Any]:
        top_k = max(1, min(int(top_k), 5))

        try:
            query_vec = embedder.embed_query(query)
            hits = search(query_vec, top_k=top_k)
        except Exception as e:
            raise ToolExecutionError(f"Knowledge search failed: {e}") from e

        if not hits:
            return {
                "query": query,
                "count": 0,
                "results": [],
                "note": (
                    "No documents found in the knowledge base. "
                    "Do NOT invent an answer."
                ),
            }

        results = [
            {
                "source": h["document_name"],
                "chunk_index": h["chunk_index"],
                "text": h["text"][:800],
                "relevance": round(1.0 - h["distance"], 3),
            }
            for h in hits
        ]

        return {
            "query": query,
            "count": len(results),
            "results": results,
            "note": (
                "These excerpts come from the user's uploaded documents. "
                "Quote them verbatim. Cite the source filename when answering."
            ),
        }