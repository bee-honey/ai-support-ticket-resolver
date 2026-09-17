"""Chunking layer shared by every source type.

A ticket (or doc) is not assumed to always be one chunk: short documents are
kept whole, long ones are split. Every chunk retains the parent document's
full metadata (ticket_id, component, status, resolved_date, source_type,
source_file, ...) plus a `chunk_index`, so citations and metadata filtering
keep working after chunking.

Chunk size/overlap are configurable and default to `app.config.settings`
values -- they are reasonable starting points, not a tuned/optimal strategy.
Chunking strategy is expected to be revisited during evaluation.
"""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings
from app.models.schemas import Document

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            if len(doc.page_content) <= self.chunk_size:
                chunks.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "chunk_index": 0},
                    )
                )
                continue

            for i, chunk_text in enumerate(self._splitter.split_text(doc.page_content)):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**doc.metadata, "chunk_index": i},
                    )
                )

        logger.info(
            "Chunked %d document(s) into %d chunk(s) (chunk_size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
