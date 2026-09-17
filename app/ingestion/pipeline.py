"""Orchestrates: pick loader by extension -> load -> chunk -> embed -> persist.

This is the one place that knows about all four loaders. Everything else
(chunking, embeddings, vector store) is source-agnostic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.embeddings.service import EmbeddingService
from app.ingestion.base import BaseDocumentLoader, IngestionError
from app.ingestion.chunker import ChunkingService
from app.ingestion.csv_loader import TicketCSVLoader
from app.ingestion.mapping import CSVFieldMapping
from app.ingestion.markdown_loader import MarkdownDocumentLoader
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.text_loader import TextDocumentLoader
from app.vectorstore.chroma_store import VectorStoreService

logger = logging.getLogger(__name__)

LOADER_BY_EXTENSION: dict[str, type[BaseDocumentLoader]] = {
    ".csv": TicketCSVLoader,
    ".pdf": PDFDocumentLoader,
    ".md": MarkdownDocumentLoader,
    ".markdown": MarkdownDocumentLoader,
    ".txt": TextDocumentLoader,
}


@dataclass
class IngestionSummary:
    source: str
    documents_loaded: int
    chunks_written: int


class IngestionPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        chunking_service: ChunkingService | None = None,
        csv_mapping: CSVFieldMapping | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.chunking_service = chunking_service or ChunkingService()
        self.csv_mapping = csv_mapping

    def _get_loader(self, source: str) -> BaseDocumentLoader:
        ext = Path(source).suffix.lower()
        loader_cls = LOADER_BY_EXTENSION.get(ext)
        if loader_cls is None:
            raise IngestionError(
                f"Unsupported source type '{ext}' for {source}. "
                f"Supported extensions: {sorted(LOADER_BY_EXTENSION)}"
            )
        if loader_cls is TicketCSVLoader:
            return TicketCSVLoader(mapping=self.csv_mapping)
        return loader_cls()

    def ingest(self, source: str, reset_source: bool = False) -> IngestionSummary:
        loader = self._get_loader(source)
        documents = loader.load(source)

        if reset_source:
            self.vector_store.delete_by_source(Path(source).name)

        chunks = self.chunking_service.chunk_documents(documents)
        written = self.vector_store.add_documents(chunks, self.embedding_service)

        logger.info(
            "Ingested %s: %d document(s) -> %d chunk(s) written",
            source,
            len(documents),
            written,
        )
        return IngestionSummary(source=source, documents_loaded=len(documents), chunks_written=written)
