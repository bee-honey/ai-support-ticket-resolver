"""Retrieval service: turns a support problem description into relevant chunks.

This is the seam meant to survive Phase 2 unchanged -- it can be called from
Streamlit today, from a FastAPI endpoint tomorrow, or wrapped as a
LangGraph tool/node later, without touching the vector store or embedding
code underneath it.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.embeddings.service import EmbeddingService
from app.models.schemas import RetrievedChunk
from app.vectorstore.chroma_store import VectorStoreService


class Retriever:
    def __init__(
        self,
        vector_store: VectorStoreService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.vector_store = vector_store or VectorStoreService()
        self.embedding_service = embedding_service or EmbeddingService()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks relevant to `query`.

        `filters` is an optional exact-match metadata filter (e.g.
        `{"component": "docker", "status": "Resolved"}`); no filter values
        are hard-coded here, callers decide what to filter on.
        """
        top_k = k or get_settings().default_top_k
        return self.vector_store.similarity_search_with_filter(
            query, self.embedding_service, k=top_k, filters=filters
        )

    def retrieve_many(
        self,
        queries: list[str],
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Same as `retrieve`, but for several query variants at once -- one embedding
        call and one Chroma call cover all of them, instead of one round trip per
        query. Used by `RAGService` to search the original question and a rewritten
        variant together (see `_generate_search_queries`)."""
        top_k = k or get_settings().default_top_k
        return self.vector_store.similarity_search_many_with_filter(
            queries, self.embedding_service, k=top_k, filters=filters
        )
