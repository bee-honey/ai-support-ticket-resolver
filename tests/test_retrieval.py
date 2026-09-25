"""Retrieval + metadata filtering, and vector store idempotency.

Uses a real (temp, on-disk) ChromaDB instance with the FakeEmbeddingService
fixture, so no OpenAI API calls are made.
"""

from __future__ import annotations

from app.models.schemas import Document
from app.retrieval.retriever import Retriever


def _seed(vector_store, fake_embedding_service, docs):
    vector_store.add_documents(docs, fake_embedding_service)
    return Retriever(vector_store=vector_store, embedding_service=fake_embedding_service)


def test_retrieve_returns_all_relevant_chunks(vector_store, fake_embedding_service):
    docs = [
        Document(
            page_content="Docker daemon socket stale after restart",
            metadata={"ticket_id": "MESOS-1001", "component": "docker", "status": "Resolved", "chunk_index": 0, "source_file": "s.csv"},
        ),
        Document(
            page_content="Network partition prevents re-registration",
            metadata={"ticket_id": "MESOS-1002", "component": "networking", "status": "Open", "chunk_index": 0, "source_file": "s.csv"},
        ),
    ]
    retriever = _seed(vector_store, fake_embedding_service, docs)
    results = retriever.retrieve("docker issue", k=5)
    assert {r.metadata["ticket_id"] for r in results} == {"MESOS-1001", "MESOS-1002"}
    assert all(r.score is not None for r in results)


def test_single_field_metadata_filter_restricts_results(vector_store, fake_embedding_service):
    docs = [
        Document(page_content="Docker daemon socket stale", metadata={"ticket_id": "MESOS-1001", "component": "docker", "status": "Resolved", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="Network partition issue", metadata={"ticket_id": "MESOS-1002", "component": "networking", "status": "Open", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="Docker fd leak", metadata={"ticket_id": "MESOS-1006", "component": "docker", "status": "Resolved", "chunk_index": 0, "source_file": "s.csv"}),
    ]
    retriever = _seed(vector_store, fake_embedding_service, docs)
    results = retriever.retrieve("issue", k=10, filters={"component": "docker"})
    assert {r.metadata["ticket_id"] for r in results} == {"MESOS-1001", "MESOS-1006"}


def test_multi_field_metadata_filter(vector_store, fake_embedding_service):
    docs = [
        Document(page_content="Docker daemon socket stale", metadata={"ticket_id": "MESOS-1001", "component": "docker", "status": "Resolved", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="Docker open bug", metadata={"ticket_id": "MESOS-1099", "component": "docker", "status": "Open", "chunk_index": 0, "source_file": "s.csv"}),
    ]
    retriever = _seed(vector_store, fake_embedding_service, docs)
    results = retriever.retrieve("docker", k=10, filters={"component": "docker", "status": "Resolved"})
    assert {r.metadata["ticket_id"] for r in results} == {"MESOS-1001"}


def test_add_documents_is_idempotent(vector_store, fake_embedding_service):
    docs = [Document(page_content="Docker daemon socket stale", metadata={"ticket_id": "MESOS-1001", "component": "docker", "chunk_index": 0, "source_file": "s.csv"})]
    vector_store.add_documents(docs, fake_embedding_service)
    vector_store.add_documents(docs, fake_embedding_service)
    assert vector_store.collection_info()["count"] == 1


def test_delete_by_source_removes_only_that_source(vector_store, fake_embedding_service):
    docs_a = [Document(page_content="A", metadata={"ticket_id": "T-A", "chunk_index": 0, "source_file": "a.csv"})]
    docs_b = [Document(page_content="B", metadata={"ticket_id": "T-B", "chunk_index": 0, "source_file": "b.csv"})]
    vector_store.add_documents(docs_a, fake_embedding_service)
    vector_store.add_documents(docs_b, fake_embedding_service)
    vector_store.delete_by_source("a.csv")
    remaining = vector_store.similarity_search("anything", fake_embedding_service, k=10)
    assert {r.metadata["ticket_id"] for r in remaining} == {"T-B"}


def test_empty_collection_returns_no_results(vector_store, fake_embedding_service):
    results = vector_store.similarity_search("anything", fake_embedding_service, k=5)
    assert results == []


def test_get_chunks_returns_full_text_matching_a_filter(vector_store, fake_embedding_service):
    docs = [
        Document(page_content="chunk 0 text", metadata={"ticket_id": "MESOS-1", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="chunk 1 text", metadata={"ticket_id": "MESOS-1", "chunk_index": 1, "source_file": "s.csv"}),
        Document(page_content="other ticket", metadata={"ticket_id": "MESOS-2", "chunk_index": 0, "source_file": "s.csv"}),
    ]
    vector_store.add_documents(docs, fake_embedding_service)
    chunks = vector_store.get_chunks(filters={"ticket_id": "MESOS-1"})
    assert {c.text for c in chunks} == {"chunk 0 text", "chunk 1 text"}
    assert vector_store.get_chunks(filters={"chunk_index": 0}) and len(vector_store.get_chunks(filters={"chunk_index": 0})) == 2


def test_component_tag_index_splits_semicolon_joined_components(vector_store, fake_embedding_service):
    docs = [
        Document(page_content="a", metadata={"ticket_id": "MESOS-1", "component": "docker;agent", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="b", metadata={"ticket_id": "MESOS-2", "component": "docker", "chunk_index": 0, "source_file": "s.csv"}),
        Document(page_content="c", metadata={"ticket_id": "MESOS-3", "component": "networking", "chunk_index": 0, "source_file": "s.csv"}),
    ]
    vector_store.add_documents(docs, fake_embedding_service)
    index = vector_store.component_tag_index()
    assert set(index["docker"]) == {"docker;agent", "docker"}  # tag "docker" maps back to both raw strings
    assert index["agent"] == ["docker;agent"]
    assert index["networking"] == ["networking"]
