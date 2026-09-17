"""CSV ticket ingestion.

Converts rows of a ticket CSV into `Document`s, keeping a strict separation
between the text that gets embedded (semantic fields: summary, description,
comments, resolution) and the fields stored purely as Chroma metadata
(ticket_id, component, status, issue_type, created_date, resolved_date).

Column names are resolved entirely through a `CSVFieldMapping`
(app/ingestion/mapping.py) -- nothing here hard-codes the real Mesos export's
column names, so a renamed dataset only requires a new mapping, not a code
change.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.ingestion.base import BaseDocumentLoader, IngestionError, clean_text
from app.ingestion.mapping import CSVFieldMapping, default_ticket_mapping
from app.models.schemas import Document

logger = logging.getLogger(__name__)

# Logical semantic field -> human-readable section heading in the embedded
# document text. Any semantic field without an explicit label here falls
# back to a title-cased version of its logical name.
SECTION_LABELS: dict[str, str] = {
    "summary": "Title",
    "description": "Problem",
    "comments": "Discussion",
    "resolution": "Resolution",
}


def build_ticket_document_text(semantic_values: dict[str, str | None]) -> str:
    """Build the embedded document text from cleaned semantic field values.

    Only non-empty sections are included, so a ticket with no comments yet
    does not embed an empty "Discussion:" section (and never embeds the
    literal text "nan").
    """
    parts: list[str] = []
    for logical_name, value in semantic_values.items():
        if not value:
            continue
        label = SECTION_LABELS.get(logical_name, logical_name.replace("_", " ").title())
        parts.append(f"{label}:\n{value}")
    return "\n\n".join(parts)


class TicketCSVLoader(BaseDocumentLoader):
    """Loads a ticket CSV into Documents using a configurable field mapping."""

    source_type = "csv"

    def __init__(self, mapping: CSVFieldMapping | None = None):
        self.mapping = mapping or default_ticket_mapping()

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"CSV source not found: {source}")

        df = pd.read_csv(path, dtype=str, keep_default_na=True)

        logical_to_column = self.mapping.all_logical_to_column()
        missing_columns = {
            logical: column
            for logical, column in logical_to_column.items()
            if logical in self.mapping.required_logical_fields and column not in df.columns
        }
        if missing_columns:
            raise IngestionError(
                "CSV is missing required column(s) for mapped field(s): "
                f"{missing_columns}. Available columns: {list(df.columns)}. "
                "Update the CSVFieldMapping to match this dataset."
            )

        documents: list[Document] = []
        skipped = 0

        for row_index, row in df.iterrows():
            semantic_values: dict[str, str | None] = {}
            for logical_name, column in self.mapping.semantic_fields.items():
                semantic_values[logical_name] = clean_text(row.get(column)) if column in df.columns else None

            metadata_values: dict[str, str | None] = {}
            for logical_name, column in self.mapping.metadata_fields.items():
                metadata_values[logical_name] = clean_text(row.get(column)) if column in df.columns else None

            missing_required = [
                name
                for name in self.mapping.required_logical_fields
                if not (semantic_values.get(name) or metadata_values.get(name))
            ]
            if missing_required:
                logger.warning(
                    "Skipping malformed row %s in %s: missing required field(s) %s",
                    row_index,
                    path.name,
                    missing_required,
                )
                skipped += 1
                continue

            text = build_ticket_document_text(semantic_values)

            metadata: dict[str, str] = {k: v for k, v in metadata_values.items() if v is not None}
            metadata["source_type"] = self.source_type
            metadata["source_file"] = path.name
            metadata["summary"] = semantic_values.get("summary") or ""

            documents.append(Document(page_content=text, metadata=metadata))

        logger.info(
            "Loaded %d ticket(s) from %s (%d skipped as malformed)",
            len(documents),
            path.name,
            skipped,
        )
        return documents
