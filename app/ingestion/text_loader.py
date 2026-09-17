"""Plain-text ingestion for FAQs / notes / misc docs."""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.base import BaseDocumentLoader, IngestionError
from app.models.schemas import Document

logger = logging.getLogger(__name__)


class TextDocumentLoader(BaseDocumentLoader):
    """Loads a plain-text file into a single Document (chunking happens downstream)."""

    source_type = "text"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"Text source not found: {source}")

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            logger.warning("Text file %s is empty", path.name)
            return []

        metadata = {
            "source_type": self.source_type,
            "source_file": path.name,
        }
        logger.info("Loaded text file %s", path.name)
        return [Document(page_content=text, metadata=metadata)]
