"""Configurable CSV column mapping for ticket ingestion.

The real Mesos export is not available yet and its exact column names may
change. `TicketCSVLoader` never hard-codes a column name -- it always goes
through a `CSVFieldMapping`, so when the real CSV arrives you only need to
adjust this mapping (inline, via a JSON file, or via env var), not the
ingestion code itself. See README "Adjusting for the real dataset".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

# Logical name -> CSV column name, for the fields that make up the embedded
# document text (semantic content).
DEFAULT_SEMANTIC_FIELDS: dict[str, str] = {
    "summary": "summary",
    "description": "description",
    "comments": "comments",
    "resolution": "resolution",
}

# Logical name -> CSV column name, for fields stored as Chroma metadata
# (filtering / citations / display) rather than embedded.
DEFAULT_METADATA_FIELDS: dict[str, str] = {
    "ticket_id": "ticket_id",
    "component": "component",
    "status": "status",
    "issue_type": "issue_type",
    "created_date": "created_date",
    "resolved_date": "resolved_date",
}

# Logical names (from semantic + metadata fields above) that must resolve to
# a present column AND a non-empty value per row, or the row is skipped.
DEFAULT_REQUIRED_LOGICAL_FIELDS: tuple[str, ...] = ("ticket_id", "summary")


@dataclass(frozen=True)
class CSVFieldMapping:
    """Maps logical ticket fields to actual CSV column names.

    `semantic_fields` values are concatenated (in dict order) into the text
    that gets embedded. `metadata_fields` values are stored as Chroma
    metadata and are never embedded.
    """

    semantic_fields: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SEMANTIC_FIELDS))
    metadata_fields: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_METADATA_FIELDS))
    required_logical_fields: tuple[str, ...] = DEFAULT_REQUIRED_LOGICAL_FIELDS

    def all_logical_to_column(self) -> dict[str, str]:
        return {**self.semantic_fields, **self.metadata_fields}

    def with_overrides(
        self,
        semantic_fields: dict[str, str] | None = None,
        metadata_fields: dict[str, str] | None = None,
        required_logical_fields: tuple[str, ...] | None = None,
    ) -> "CSVFieldMapping":
        """Return a copy with only the given logical->column names replaced.

        Useful when the real dataset renames a handful of columns, e.g.::

            default_ticket_mapping().with_overrides(
                metadata_fields={"ticket_id": "key", "component": "components",
                                  "resolved_date": "resolved"}
            )
        """
        merged_semantic = {**self.semantic_fields, **(semantic_fields or {})}
        merged_metadata = {**self.metadata_fields, **(metadata_fields or {})}
        return replace(
            self,
            semantic_fields=merged_semantic,
            metadata_fields=merged_metadata,
            required_logical_fields=required_logical_fields or self.required_logical_fields,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CSVFieldMapping":
        """Load a mapping override file, e.g.::

            {
              "semantic_fields": {"summary": "summary"},
              "metadata_fields": {"ticket_id": "key", "component": "components"},
              "required_logical_fields": ["ticket_id", "summary"]
            }

        Any field omitted from the file falls back to the default mapping.
        """
        data = json.loads(Path(path).read_text())
        return default_ticket_mapping().with_overrides(
            semantic_fields=data.get("semantic_fields"),
            metadata_fields=data.get("metadata_fields"),
            required_logical_fields=(
                tuple(data["required_logical_fields"]) if "required_logical_fields" in data else None
            ),
        )


def default_ticket_mapping() -> CSVFieldMapping:
    return CSVFieldMapping()
