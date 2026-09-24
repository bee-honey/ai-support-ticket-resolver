"""ensure_ingested(): the auto-ingest-if-empty check used on app boot. No real I/O -- both
the vector store and the pipeline it drives are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.ingestion.bootstrap import DEFAULT_MAPPING, DEFAULT_SOURCE, ensure_ingested
from app.ingestion.pipeline import IngestionSummary


def _vector_store(count: int) -> MagicMock:
    store = MagicMock()
    store.collection_info.return_value = {"name": "support_tickets", "count": count, "persist_dir": "chroma_db"}
    return store


def test_does_nothing_when_the_collection_already_has_data():
    store = _vector_store(count=8034)
    with patch("app.ingestion.bootstrap.IngestionPipeline") as pipeline_cls:
        result = ensure_ingested(store)
    assert result is None
    pipeline_cls.assert_not_called()


def test_ingests_the_default_dataset_when_the_collection_is_empty():
    store = _vector_store(count=0)
    summary = IngestionSummary(source=DEFAULT_SOURCE, documents_loaded=500, chunks_written=8034)
    with patch("app.ingestion.bootstrap.CSVFieldMapping") as mapping_cls, \
         patch("app.ingestion.bootstrap.IngestionPipeline") as pipeline_cls:
        pipeline_cls.return_value.ingest.return_value = summary
        result = ensure_ingested(store)

    mapping_cls.from_json.assert_called_once_with(DEFAULT_MAPPING)
    pipeline_cls.assert_called_once_with(vector_store=store, csv_mapping=mapping_cls.from_json.return_value)
    pipeline_cls.return_value.ingest.assert_called_once_with(DEFAULT_SOURCE)
    assert result is summary


def test_custom_source_and_mapping_are_honoured():
    store = _vector_store(count=0)
    with patch("app.ingestion.bootstrap.CSVFieldMapping") as mapping_cls, \
         patch("app.ingestion.bootstrap.IngestionPipeline") as pipeline_cls:
        ensure_ingested(store, source="data/other.csv", mapping_path="config/other.json")

    mapping_cls.from_json.assert_called_once_with("config/other.json")
    pipeline_cls.return_value.ingest.assert_called_once_with("data/other.csv")
