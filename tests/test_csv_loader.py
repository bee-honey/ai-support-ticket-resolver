"""CSV -> Document conversion: semantic/metadata separation, NaN handling,
malformed rows, and configurable column mapping."""

from __future__ import annotations

import csv

import pytest

from app.ingestion.base import IngestionError
from app.ingestion.csv_loader import TicketCSVLoader, build_ticket_document_text
from app.ingestion.mapping import CSVFieldMapping

FULL_HEADER = [
    "ticket_id",
    "summary",
    "description",
    "comments",
    "resolution",
    "component",
    "status",
    "issue_type",
    "created_date",
    "resolved_date",
]


def write_csv(tmp_path, rows: list[dict], header: list[str]) -> str:
    path = tmp_path / "tickets.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_loads_sample_csv(sample_csv_path):
    docs = TicketCSVLoader().load(sample_csv_path)
    assert len(docs) == 10
    doc = next(d for d in docs if d.metadata["ticket_id"] == "MESOS-1001")
    assert "Title:\nDocker executor fails during startup" in doc.page_content
    assert "Resolution:\nRestarting the Docker daemon" in doc.page_content


def test_semantic_content_not_duplicated_into_metadata(sample_csv_path):
    doc = next(
        d for d in TicketCSVLoader().load(sample_csv_path) if d.metadata["ticket_id"] == "MESOS-1001"
    )
    assert "description" not in doc.metadata
    assert "comments" not in doc.metadata
    assert "resolution" not in doc.metadata
    assert doc.metadata["component"] == "docker"
    assert doc.metadata["status"] == "Resolved"


def test_nan_and_empty_fields_are_excluded_not_stringified(sample_csv_path):
    # MESOS-1002 has empty comments/resolution/resolved_date in the sample CSV.
    doc = next(
        d for d in TicketCSVLoader().load(sample_csv_path) if d.metadata["ticket_id"] == "MESOS-1002"
    )
    assert "nan" not in doc.page_content.lower()
    assert "Resolution:" not in doc.page_content
    assert "resolved_date" not in doc.metadata


def test_missing_required_column_raises_ingestion_error(tmp_path):
    path = write_csv(tmp_path, rows=[{"summary": "x", "description": "y"}], header=["summary", "description"])
    with pytest.raises(IngestionError):
        TicketCSVLoader().load(path)


def test_malformed_row_is_skipped_not_fatal(tmp_path):
    rows = [
        {
            "ticket_id": "T-1",
            "summary": "Good ticket",
            "description": "d",
            "comments": "",
            "resolution": "",
            "component": "x",
            "status": "Open",
            "issue_type": "Bug",
            "created_date": "2024-01-01",
            "resolved_date": "",
        },
        {
            "ticket_id": "",  # missing required field -> skipped, not fatal
            "summary": "",
            "description": "d2",
            "comments": "",
            "resolution": "",
            "component": "x",
            "status": "Open",
            "issue_type": "Bug",
            "created_date": "2024-01-02",
            "resolved_date": "",
        },
    ]
    path = write_csv(tmp_path, rows, FULL_HEADER)
    docs = TicketCSVLoader().load(path)
    assert len(docs) == 1
    assert docs[0].metadata["ticket_id"] == "T-1"


def test_custom_mapping_handles_renamed_columns(tmp_path):
    header = ["key", "summary", "description", "comments", "resolution", "components", "status", "issue_type", "created_date", "resolved"]
    rows = [
        {
            "key": "T-9",
            "summary": "Renamed columns",
            "description": "d",
            "comments": "",
            "resolution": "",
            "components": "docker",
            "status": "Open",
            "issue_type": "Bug",
            "created_date": "2024-01-01",
            "resolved": "",
        }
    ]
    path = write_csv(tmp_path, rows, header)
    mapping = CSVFieldMapping().with_overrides(
        metadata_fields={"ticket_id": "key", "component": "components", "resolved_date": "resolved"}
    )
    docs = TicketCSVLoader(mapping=mapping).load(path)
    assert len(docs) == 1
    assert docs[0].metadata["ticket_id"] == "T-9"
    assert docs[0].metadata["component"] == "docker"


def test_build_ticket_document_text_skips_empty_sections():
    text = build_ticket_document_text({"summary": "S", "description": None, "comments": "", "resolution": "R"})
    assert "Problem:" not in text
    assert "Discussion:" not in text
    assert text.startswith("Title:\nS")
    assert text.endswith("Resolution:\nR")
