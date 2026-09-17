"""Loader abstraction shared by every document source (CSV, PDF, Markdown, TXT).

Each loader converts one source into a list of `Document`s carrying
`page_content` + `metadata`. Everything downstream (chunking, embedding,
storage) works the same regardless of which loader produced the documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from app.models.schemas import Document


class IngestionError(Exception):
    """Raised when a source cannot be loaded at all (e.g. missing required columns/file)."""


class BaseDocumentLoader(ABC):
    """Common interface every source-specific loader implements."""

    source_type: str = "unknown"

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load `source` and return a list of Documents.

        Implementations should raise `IngestionError` for structural problems
        (missing file, missing required columns) and should skip + log
        individual malformed records rather than failing the whole load.
        """
        raise NotImplementedError


def clean_text(value: Any) -> str | None:
    """Normalize a raw cell/field value to a stripped string, or None.

    Treats None, NaN/NaT, and the literal strings "nan"/"none"/"null"
    (case-insensitive) as absent, so they never leak into embedded text or
    Chroma metadata as junk text.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return text
