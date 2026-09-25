"""Look up one ticket's fields straight from its source CSV, unchunked.

Used by the Tickets browser's detail view: `views/tickets.py` shows the
retrieved-chunk text (what the retriever actually works with) separately from
this -- the ticket exactly as it exists in the source data, with
summary/description/comments/resolution kept as whole fields rather than
split, joined, and re-split into overlapping chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ingestion.base import clean_text
from app.ingestion.mapping import CSVFieldMapping, default_ticket_mapping

# source_file (as stored in chunk metadata, e.g. "mesos_scoped.csv") -> (path, mapping
# JSON path or None for the default mapping). Add an entry whenever a new CSV with a
# non-default column mapping is ingested -- see README "Adjusting for the Real Mesos CSV".
KNOWN_SOURCES: dict[str, tuple[str, str | None]] = {
    "mesos_scoped.csv": ("data/mesos_scoped.csv", "config/mesos_mapping.json"),
    "sample_tickets.csv": ("data/sample/sample_tickets.csv", None),
}

# Matches a comment entry's leading "[2011-09-30T22:36:41.291+0000] " marker, at the
# start of a line -- this corpus's convention for separating discussion entries. A
# comment's own text may itself span multiple paragraphs before the next marker.
_COMMENT_MARKER = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{4})\]\s?", re.MULTILINE)


@dataclass
class CommentEntry:
    timestamp: str | None  # raw ISO 8601, or None if this entry had no leading marker
    text: str


@dataclass
class TicketText:
    summary: str
    description: str
    comments: list[CommentEntry]
    resolution: str


def parse_comments(raw: str | None) -> list[CommentEntry]:
    """Split a raw comments blob into individual entries by their timestamp markers.

    Falls back to one untimestamped entry holding the whole blob if the corpus's
    marker convention isn't present, so free-form discussion still displays.
    """
    if not raw or not raw.strip():
        return []
    matches = list(_COMMENT_MARKER.finditer(raw))
    if not matches:
        return [CommentEntry(timestamp=None, text=raw.strip())]
    entries = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        text = raw[start:end].strip()
        if text:
            entries.append(CommentEntry(timestamp=match.group(1), text=text))
    return entries


def _resolve_source(source_file: str) -> tuple[str, CSVFieldMapping]:
    path, mapping_path = KNOWN_SOURCES.get(source_file, (f"data/{source_file}", None))
    mapping = CSVFieldMapping.from_json(mapping_path) if mapping_path else default_ticket_mapping()
    return path, mapping


def load_ticket_text(ticket_id: str, source_file: str) -> TicketText | None:
    """The one matching row's semantic fields, read fresh from `source_file`'s CSV.

    None if the source file is missing or the ticket isn't found in it (e.g. it
    came from a non-CSV loader) -- callers should fall back to the chunked text
    from Chroma in that case, not treat this as a hard failure.
    """
    path, mapping = _resolve_source(source_file)
    if not Path(path).exists():
        return None

    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    id_column = mapping.metadata_fields.get("ticket_id")
    if id_column not in df.columns:
        return None

    matches = df[df[id_column] == ticket_id]
    if matches.empty:
        return None
    row = matches.iloc[0]

    def field(logical_name: str) -> str:
        column = mapping.semantic_fields.get(logical_name)
        return (clean_text(row.get(column)) if column and column in df.columns else None) or ""

    return TicketText(
        summary=field("summary"),
        description=field("description"),
        comments=parse_comments(field("comments") or None),
        resolution=field("resolution"),
    )
