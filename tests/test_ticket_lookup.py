"""load_ticket_text() / parse_comments(): reading a single ticket's raw fields
straight from a CSV, unchunked, and splitting its comments blob into entries."""

from __future__ import annotations

import csv as csv_module

import pytest

from app.ingestion.ticket_lookup import CommentEntry, load_ticket_text, parse_comments


@pytest.fixture
def default_mapping_csv(tmp_path):
    """A CSV using the DEFAULT column names (no mapping override needed)."""
    path = tmp_path / "tickets.csv"
    with path.open("w", newline="") as f:
        writer = csv_module.DictWriter(
            f, fieldnames=["ticket_id", "summary", "description", "comments", "resolution"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "ticket_id": "T-1",
                "summary": "Docker daemon crash",
                "description": "Crashes on boot after upgrade.",
                "comments": (
                    "[2020-01-01T00:00:00.000+0000] First look: stale socket.\n\n"
                    "Second paragraph of the same comment, still no marker.\n\n"
                    "[2020-01-02T09:30:00.000+0000] Confirmed fix works."
                ),
                "resolution": "Fixed by restarting the daemon service.",
            }
        )
        writer.writerow({"ticket_id": "T-2", "summary": "No comments yet", "description": "d", "comments": "", "resolution": ""})
    return path


def test_loads_the_matching_row_fields(monkeypatch, default_mapping_csv):
    from app.ingestion import ticket_lookup

    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "tickets.csv", (str(default_mapping_csv), None))
    text = load_ticket_text("T-1", "tickets.csv")
    assert text.summary == "Docker daemon crash"
    assert text.description == "Crashes on boot after upgrade."
    assert text.resolution == "Fixed by restarting the daemon service."


def test_comments_split_by_timestamp_marker_not_by_blank_lines(monkeypatch, default_mapping_csv):
    from app.ingestion import ticket_lookup

    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "tickets.csv", (str(default_mapping_csv), None))
    text = load_ticket_text("T-1", "tickets.csv")
    assert len(text.comments) == 2  # NOT 3 -- the blank-line-separated paragraph stays in comment 1
    assert text.comments[0].timestamp == "2020-01-01T00:00:00.000+0000"
    assert "First look" in text.comments[0].text and "Second paragraph" in text.comments[0].text
    assert text.comments[1].timestamp == "2020-01-02T09:30:00.000+0000"
    assert text.comments[1].text == "Confirmed fix works."


def test_ticket_with_no_comments_returns_an_empty_list(monkeypatch, default_mapping_csv):
    from app.ingestion import ticket_lookup

    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "tickets.csv", (str(default_mapping_csv), None))
    text = load_ticket_text("T-2", "tickets.csv")
    assert text.comments == []


def test_unknown_ticket_id_returns_none(monkeypatch, default_mapping_csv):
    from app.ingestion import ticket_lookup

    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "tickets.csv", (str(default_mapping_csv), None))
    assert load_ticket_text("T-DOES-NOT-EXIST", "tickets.csv") is None


def test_unknown_source_file_returns_none():
    assert load_ticket_text("T-1", "a-source-nobody-registered.csv") is None


def test_mapped_column_names_are_respected(tmp_path, monkeypatch):
    # mirrors the real mesos_scoped.csv convention: ticket_id column is actually "key"
    path = tmp_path / "raw_export.csv"
    with path.open("w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["key", "summary", "description", "all_comments", "resolution"])
        writer.writeheader()
        writer.writerow({"key": "PROJ-9", "summary": "s", "description": "d", "all_comments": "", "resolution": "r"})

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text('{"metadata_fields": {"ticket_id": "key"}, "semantic_fields": {"comments": "all_comments"}}')

    from app.ingestion import ticket_lookup

    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "raw_export.csv", (str(path), str(mapping_path)))
    text = load_ticket_text("PROJ-9", "raw_export.csv")
    assert text.summary == "s" and text.resolution == "r"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("just free text, no markers at all", [CommentEntry(timestamp=None, text="just free text, no markers at all")]),
        ("[2020-01-01T00:00:00.000+0000] one entry", [CommentEntry(timestamp="2020-01-01T00:00:00.000+0000", text="one entry")]),
    ],
)
def test_parse_comments_edge_cases(raw, expected):
    assert parse_comments(raw) == expected
