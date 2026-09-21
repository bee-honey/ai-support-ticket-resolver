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
    # Timing (seconds) and token usage for the response snapshot in the UI.
    # Token counts are None when the provider doesn't report them or when no
    # LLM call was made (e.g. the no-evidence short-circuit).
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    # The raw retrieved chunks (before per-ticket dedupe) -- the eval framework
    # needs the exact evidence the LLM saw in order to judge it.
    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)
