"""Tickets page, driven headlessly with AppTest against a real (temp, on-disk) Chroma
instance seeded with FakeEmbeddingService -- exercises the actual data loading and
filtering logic, not just the UI shell. No OpenAI calls.

`VectorStoreService()` is patched to return the pre-seeded instance directly (rather
than redirecting via CHROMA_PERSIST_DIR env vars) because app.config.settings.get_settings()
is @lru_cache'd for the whole test session -- an env var set here wouldn't be seen by
code that already called get_settings() earlier in the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.models.schemas import Document
from app.vectorstore.chroma_store import VectorStoreService
from tests.conftest import FakeEmbeddingService

PAGE = str(Path(__file__).resolve().parent.parent / "ui" / "views" / "tickets.py")

DOCKER_TICKET = [
    Document(
        page_content="Title: Docker daemon crash\n\nProblem: crashes on boot",
        metadata={
            "ticket_id": "MESOS-1", "summary": "Docker daemon crash", "component": "docker;agent",
            "status": "Resolved", "issue_type": "Bug", "created_date": "2020-01-01T00:00:00.000+0000",
            "resolved_date": "2020-01-05T00:00:00.000+0000", "source_file": "s.csv", "chunk_index": 0,
        },
    ),
    Document(
        page_content="Discussion: turned out to be a stale socket after restart",
        metadata={"ticket_id": "MESOS-1", "component": "docker;agent", "status": "Resolved", "chunk_index": 1, "source_file": "s.csv"},
    ),
]
NETWORK_TICKET = [
    Document(
        page_content="Title: Network timeout\n\nProblem: agent loses connection to master",
        metadata={
            "ticket_id": "MESOS-2", "summary": "Network timeout", "component": "networking",
            "status": "Open", "issue_type": "Bug", "source_file": "s.csv", "chunk_index": 0,
        },
    ),
]


@pytest.fixture
def seeded_store(tmp_path) -> VectorStoreService:
    store = VectorStoreService(persist_dir=str(tmp_path / "chroma"), collection_name="test_tickets")
    store.add_documents(DOCKER_TICKET + NETWORK_TICKET, FakeEmbeddingService())
    return store


@pytest.fixture(autouse=True)
def patch_vector_store(seeded_store, monkeypatch):
    import app.vectorstore.chroma_store as chroma_store
    import streamlit as st

    monkeypatch.setattr(chroma_store, "VectorStoreService", lambda: seeded_store)
    st.cache_resource.clear()
    st.cache_data.clear()


def _page(ticket: str | None = None) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=30)
    if ticket:
        at.query_params["ticket"] = ticket
    return at.run()


def _table(at: AppTest):
    return next(d for d in at.dataframe if d.key == "ticket_table")


def test_list_shows_every_ingested_ticket_once():
    at = _page()
    assert not at.exception
    rows = _table(at).value
    assert set(rows["ticket_id"]) == {"MESOS-1", "MESOS-2"}
    assert len(rows) == 2  # one row per ticket, not one per chunk


def test_search_filters_by_id_or_summary():
    at = _page()
    next(t for t in at.text_input if t.label == "Search (id or summary)").set_value("timeout").run()
    rows = _table(at).value
    assert set(rows["ticket_id"]) == {"MESOS-2"}


def test_status_filter_restricts_to_matching_tickets():
    at = _page()
    next(s for s in at.selectbox if s.label == "Status").select("Open").run()
    rows = _table(at).value
    assert set(rows["ticket_id"]) == {"MESOS-2"}


def test_component_filter_uses_the_tag_index_not_raw_strings():
    at = _page()
    next(m for m in at.multiselect if m.label == "Component").select("docker").run()
    rows = _table(at).value
    assert set(rows["ticket_id"]) == {"MESOS-1"}


def test_deep_link_jumps_straight_to_the_tickets_detail():
    at = _page(ticket="MESOS-1")
    assert not at.exception
    assert any(m.value == "### MESOS-1" for m in at.markdown)
    assert any("Docker daemon crash" in m.value for m in at.markdown)
    # both chunks are shown, in order, as separate expandable units
    labels = [e.label for e in at.expander]
    assert "Chunk 1 of 2" in labels and "Chunk 2 of 2" in labels


def test_back_button_clears_the_query_param():
    at = _page(ticket="MESOS-1")
    next(b for b in at.button if "Back to all tickets" in b.label).click().run()
    assert at.query_params.get("ticket") is None
    assert _table(at).value is not None  # the list is showing again


def test_deep_link_to_a_nonexistent_ticket_shows_an_error_not_a_crash():
    at = _page(ticket="MESOS-999")
    assert not at.exception
    assert any("No ticket found" in e.value for e in at.error)


def test_selecting_a_row_in_the_table_shows_its_detail():
    # AppTest has no click-simulation API for st.dataframe row selection; this
    # sets the same session_state shape a frontend click would produce.
    at = _page()
    at.session_state["ticket_table"] = {"selection": {"rows": [1], "columns": []}}  # MESOS-2 (sorted 2nd)
    at.run()
    assert any(m.value == "### MESOS-2" for m in at.markdown)


def test_empty_store_shows_a_helpful_message_not_a_crash(monkeypatch):
    import app.vectorstore.chroma_store as chroma_store

    empty_store = VectorStoreService(persist_dir=str(Path("/tmp/does-not-matter")), collection_name="empty")
    monkeypatch.setattr(chroma_store, "VectorStoreService", lambda: empty_store)
    at = _page()
    assert not at.exception
    assert any("No tickets are ingested yet" in i.value for i in at.info)
