"""Vector store abstraction over ChromaDB.

Nothing outside this module talks to `chromadb` directly. That keeps the
application layer (retriever, RAG service) decoupled from Chroma's specific
API, which matters if we ever need to swap vector databases later.

IDs are built deterministically from ticket_id/source_file + chunk_index, and
writes use `upsert`, so re-running ingestion on the same source updates
existing vectors in place instead of creating duplicates.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import get_settings
from app.models.schemas import Document, RetrievedChunk

logger = logging.getLogger(__name__)

# Chroma's telemetry client has a known incompatibility with recent posthog
# versions that logs a noisy (harmless) warning on every call; silence it.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)


class VectorStoreService:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Explicit cosine space so query distances can be converted to a
        # meaningful similarity score (1 - distance) in similarity_search*.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_id(document: Document) -> str:
        metadata = document.metadata
        chunk_index = metadata.get("chunk_index", 0)
        ticket_id = metadata.get("ticket_id")
        if ticket_id:
            return f"{ticket_id}::chunk::{chunk_index}"
        source_file = metadata.get("source_file", "unknown")
        return f"{source_file}::chunk::{chunk_index}"

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        # Chroma metadata values must be str/int/float/bool -- drop anything
        # else (e.g. None left over from an upstream loader).
        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: value}
        return {"$and": [{k: v} for k, v in filters.items()]}

    def add_documents(self, documents: list[Document], embedding_service) -> int:
        if not documents:
            return 0

        ids = [self._build_id(d) for d in documents]
        texts = [d.page_content for d in documents]
        metadatas = [self._sanitize_metadata(d.metadata) for d in documents]
        embeddings = embedding_service.embed_documents(texts)

        self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Upserted %d chunk(s) into collection '%s'", len(documents), self.collection_name)
        return len(documents)

    def similarity_search(self, query: str, embedding_service, k: int = 5) -> list[RetrievedChunk]:
        return self.similarity_search_with_filter(query, embedding_service, k=k, filters=None)

    def similarity_search_with_filter(
        self,
        query: str,
        embedding_service,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = embedding_service.embed_query(query)
        where = self._build_where(filters)

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[None] * len(ids)])[0]

        chunks: list[RetrievedChunk] = []
        for doc_text, metadata, distance in zip(docs, metadatas, distances):
            score = (1.0 - distance) if distance is not None else None
            chunks.append(RetrievedChunk(text=doc_text, metadata=dict(metadata), score=score))
        return chunks

    def delete_by_source(self, source_file: str) -> None:
        self._collection.delete(where={"source_file": source_file})
        logger.info("Deleted chunks with source_file='%s' from '%s'", source_file, self.collection_name)

    def collection_info(self) -> dict[str, Any]:
        return {
            "name": self.collection_name,
            "count": self._collection.count(),
            "persist_dir": self.persist_dir,
        }
