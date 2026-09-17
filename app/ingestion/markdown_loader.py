"""Markdown ingestion for runbooks / troubleshooting docs / FAQs."""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.base import BaseDocumentLoader, IngestionError
from app.models.schemas import Document

logger = logging.getLogger(__name__)


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Loads a Markdown file into a single Document (chunking happens downstream)."""

    source_type = "markdown"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"Markdown source not found: {source}")

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            logger.warning("Markdown file %s is empty", path.name)
            return []

        title = next(
            (line.lstrip("#").strip() for line in text.splitlines() if line.strip().startswith("#")),
            path.stem,
        )

        metadata = {
            "source_type": self.source_type,
            "source_file": path.name,
            "title": title,
        }
        logger.info("Loaded Markdown %s", path.name)
        return [Document(page_content=text, metadata=metadata)]
