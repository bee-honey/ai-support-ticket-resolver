"""PDF ingestion for runbooks / troubleshooting guides."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.base import BaseDocumentLoader, IngestionError
from app.models.schemas import Document

logger = logging.getLogger(__name__)


class PDFDocumentLoader(BaseDocumentLoader):
    """Loads a PDF file into a single Document (chunking happens downstream)."""

    source_type = "pdf"

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"PDF source not found: {source}")

        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # malformed PDF
            raise IngestionError(f"Failed to read PDF {source}: {exc}") from exc

        page_texts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(t.strip() for t in page_texts if t.strip())

        if not text:
            logger.warning("No extractable text found in PDF %s", path.name)
            return []

        metadata = {
            "source_type": self.source_type,
            "source_file": path.name,
            "page_count": str(len(reader.pages)),
        }
        logger.info("Loaded PDF %s (%d pages)", path.name, len(reader.pages))
        return [Document(page_content=text, metadata=metadata)]
