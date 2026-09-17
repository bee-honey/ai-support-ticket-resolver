"""Chunking: short docs stay whole, long docs split, metadata survives either way."""

from __future__ import annotations

from app.ingestion.chunker import ChunkingService
from app.models.schemas import Document


def test_short_document_is_not_split():
    docs = [Document(page_content="short text", metadata={"ticket_id": "T-1"})]
    chunks = ChunkingService(chunk_size=1000, chunk_overlap=100).chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["ticket_id"] == "T-1"


def test_long_document_is_split_into_multiple_chunks():
    long_text = "Sentence about docker containers failing to launch. " * 100
    docs = [Document(page_content=long_text, metadata={"ticket_id": "T-2", "component": "docker"})]
    chunks = ChunkingService(chunk_size=200, chunk_overlap=50).chunk_documents(docs)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


def test_chunk_metadata_preserves_all_parent_fields():
    parent_metadata = {
        "ticket_id": "T-3",
        "component": "storage",
        "status": "Resolved",
        "resolved_date": "2024-01-01",
        "source_type": "csv",
        "source_file": "f.csv",
    }
    docs = [Document(page_content="x " * 500, metadata=dict(parent_metadata))]
    chunks = ChunkingService(chunk_size=100, chunk_overlap=20).chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        for key, value in parent_metadata.items():
            assert chunk.metadata[key] == value


def test_ingestion_metadata_added_alongside_source_metadata():
    docs = [Document(page_content="short", metadata={"ticket_id": "T-4", "source_type": "csv", "source_file": "f.csv"})]
    chunks = ChunkingService().chunk_documents(docs)
    assert chunks[0].metadata["source_type"] == "csv"
    assert chunks[0].metadata["source_file"] == "f.csv"
    assert "chunk_index" in chunks[0].metadata


def test_defaults_come_from_settings_and_are_positive():
    service = ChunkingService()
    assert service.chunk_size > 0
    assert service.chunk_overlap >= 0
