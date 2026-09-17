"""Internal data structures shared across ingestion, retrieval, and RAG.

`Document` is re-exported from `langchain_core` so every layer (loaders,
chunker, vector store, retriever) speaks the same `page_content` /
`metadata` shape without us inventing a parallel type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

__all__ = ["Document", "RetrievedChunk", "RAGSource", "RAGResult"]


@dataclass
class RetrievedChunk:
    """A single chunk returned from the vector store, with its similarity score."""

    text: str
    metadata: dict[str, Any]
    score: float | None = None

    @property
    def ticket_id(self) -> str | None:
        return self.metadata.get("ticket_id")


@dataclass
class RAGSource:
    """A deduplicated, display-ready citation for the RAG answer."""

    ticket_id: str | None
    component: str | None
    status: str | None
    summary: str | None
    resolved_date: str | None
    source_file: str | None


@dataclass
class RAGResult:
    answer: str
    sources: list[RAGSource] = field(default_factory=list)
