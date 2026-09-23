"""Embedding wrapper around Ollama using nomic-embed-text.

IMPORTANT: nomic-embed-text requires task prefixes:
  - "search_document: " for documents
  - "search_query: "    for queries

Using the wrong prefix (or none) degrades retrieval quality significantly.
"""

from ollama import Client as OllamaClient

from app.core.config import settings


EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    """Embed text into vectors using Ollama (nomic-embed-text)."""

    def __init__(self, model: str = EMBED_MODEL, base_url: str | None = None):
        self.model = model
        self.client = OllamaClient(host=base_url or settings.ollama_base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of DOCUMENTS (adds 'search_document:' prefix)."""
        if not texts:
            return []
        prefixed = [DOC_PREFIX + t for t in texts]
        response = self.client.embed(model=self.model, input=prefixed)
        return response["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single QUERY (adds 'search_query:' prefix)."""
        prefixed = QUERY_PREFIX + text
        response = self.client.embed(model=self.model, input=[prefixed])
        return response["embeddings"][0]


# Singleton
embedder = Embedder()