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

APP = str(Path(__file__).resolve().parent.parent / "ui" / "resolver_support.py")

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
    # Through the full entrypoint, not the bare page file: st.page_link (used by this
    # page's own clickable ticket rows, and by cross-page citation links) needs an
    # active st.navigation() context to resolve its target.
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.switch_page("views/tickets.py")
    if ticket:
        at.query_params["ticket"] = ticket
    return at.run()


def _table(at: AppTest):
    return next(d for d in at.dataframe if d.key == "ticket_table")


def _select_row(at: AppTest, position: int) -> AppTest:
    # AppTest has no click-simulation API for st.dataframe row selection; this sets
    # the same session_state shape a frontend checkbox click would produce.
    at.session_state["ticket_table"] = {"selection": {"rows": [position], "columns": []}}
    return at.run()


def test_list_shows_every_ingested_ticket_once():
    at = _page()
    assert not at.exception
    ids = _table(at).value["ID"].tolist()
    assert ids.count("MESOS-1") == 1 and ids.count("MESOS-2") == 1  # one row per ticket, not per chunk


def test_search_filters_by_id_or_summary():
    at = _page()
    at = next(t for t in at.text_input if t.label == "Search tickets").set_value("timeout").run()
    assert _table(at).value["ID"].tolist() == ["MESOS-2"]


def test_status_filter_restricts_to_matching_tickets():
    at = _page()
    at = next(s for s in at.selectbox if s.label == "Status").select("Open").run()
    assert _table(at).value["ID"].tolist() == ["MESOS-2"]


def test_component_filter_uses_the_tag_index_not_raw_strings():
    at = _page()
    at = next(m for m in at.multiselect if m.label == "Component").select("docker").run()
    assert _table(at).value["ID"].tolist() == ["MESOS-1"]


def test_dates_in_the_list_are_human_formatted_not_raw_iso():
    at = _page()
    resolved = _table(at).value["Resolved"].tolist()
    assert "Jan 05, 2020" in resolved
    assert not any("2020-01-05T" in str(v) for v in resolved)


def test_deep_link_jumps_straight_to_the_tickets_detail_only_not_the_whole_tab():
    # the regression this covers: a citation link used to open just the ticket; a
    # prior refactor made it show the full 1151-row table above the detail too.
    at = _page(ticket="MESOS-1")
    assert not at.exception
    assert any(m.value == "### MESOS-1" for m in at.markdown)
    assert any("Docker daemon crash" in m.value for m in at.markdown)
    # both chunks are shown, in order, as separate expandable units
    labels = [e.label for e in at.expander]
    assert "Chunk 1 of 2" in labels and "Chunk 2 of 2" in labels
    with pytest.raises(StopIteration):
        _table(at)  # the table must NOT render in this focused, detail-only view
    assert any("Browse all tickets" in b.label for b in at.button)


def test_browse_all_tickets_button_switches_to_the_table_keeping_the_ticket_in_view():
    at = _page(ticket="MESOS-1")
    at = next(b for b in at.button if "Browse all tickets" in b.label).click().run()
    assert not at.exception
    assert _table(at) is not None  # now showing the table
    assert any(m.value == "### MESOS-1" for m in at.markdown)  # ...with the same ticket still shown below it


def test_selecting_a_row_shows_its_detail_below_the_table_and_syncs_the_url():
    at = _page()
    assert not any(m.value.startswith("### ") for m in at.markdown)  # nothing selected yet
    at = _select_row(at, 1)  # MESOS-2 (sorted 2nd)
    assert not at.exception
    assert any(m.value == "### MESOS-2" for m in at.markdown)
    assert _table(at) is not None  # the table is still showing, not replaced
    assert at.query_params.get("ticket") == ["MESOS-2"]  # stays a shareable deep link


def test_switching_tickets_while_browsing_stays_in_browse_mode():
    at = _page(ticket="MESOS-1")
    at = next(b for b in at.button if "Browse all tickets" in b.label).click().run()
    at = _select_row(at, 1)  # click MESOS-2 while browsing
    assert any(m.value == "### MESOS-2" for m in at.markdown)
    assert not any(m.value == "### MESOS-1" for m in at.markdown)
    assert at.query_params.get("ticket") == ["MESOS-2"]
    assert _table(at) is not None  # stays in browse mode -- table still visible


def test_a_new_deep_link_resets_out_of_browse_mode():
    # even after browsing the full table, clicking a *different* citation elsewhere
    # should land back on the focused, detail-only view for that new ticket.
    at = _page(ticket="MESOS-1")
    at = next(b for b in at.button if "Browse all tickets" in b.label).click().run()
    assert _table(at) is not None

    at.query_params["ticket"] = "MESOS-2"
    at = at.run()
    assert any(m.value == "### MESOS-2" for m in at.markdown)
    with pytest.raises(StopIteration):
        _table(at)


def test_clear_button_closes_the_detail_and_deselects_the_row():
    at = _page()
    at = _select_row(at, 0)
    assert any(m.value == "### MESOS-1" for m in at.markdown)
    at = next(b for b in at.button if "Clear" in b.label).click().run()
    assert at.query_params.get("ticket") is None
    assert not any(m.value.startswith("### ") for m in at.markdown)
    assert _table(at) is not None  # the list stays visible throughout


def test_deep_link_to_a_nonexistent_ticket_shows_an_error_not_a_crash():
    at = _page(ticket="MESOS-999")
    assert not at.exception
    assert any("No ticket found" in e.value for e in at.error)


def test_empty_store_shows_a_helpful_message_not_a_crash(tmp_path, monkeypatch):
    import app.ingestion.bootstrap as bootstrap
    import app.vectorstore.chroma_store as chroma_store

    empty_store = VectorStoreService(persist_dir=str(tmp_path / "empty_chroma"), collection_name="empty")
    monkeypatch.setattr(chroma_store, "VectorStoreService", lambda: empty_store)
    # An empty store also makes the app shell's bootstrap think nothing is ingested --
    # neutralize it here so this test doesn't trigger a real ingestion attempt.
    monkeypatch.setattr(bootstrap, "ensure_ingested", lambda *a, **k: None)
    at = _page()
    assert not at.exception
    assert any("No tickets are ingested yet" in i.value for i in at.info)


def test_detail_view_shows_csv_sourced_text_not_chunks_when_the_source_is_known(tmp_path, monkeypatch):
    """When the ticket's source CSV is registered, the detail view should show the
    whole, unchunked fields (with comments as separate nicely-formatted entries),
    not the chunked/embedded text."""
    import csv as csv_module

    import app.ingestion.ticket_lookup as ticket_lookup

    csv_path = tmp_path / "known_source.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["ticket_id", "summary", "description", "comments", "resolution"])
        writer.writeheader()
        writer.writerow(
            {
                "ticket_id": "MESOS-1", "summary": "Docker daemon crash",
                "description": "The daemon crashes on boot after the last upgrade.",
                "comments": (
                    "[2020-01-01T00:00:00.000+0000] Looks like a stale socket.\n\n"
                    "[2020-01-02T09:30:00.000+0000] Confirmed -- restarting fixed it."
                ),
                "resolution": "Restart the daemon service after upgrading.",
            }
        )
    monkeypatch.setitem(ticket_lookup.KNOWN_SOURCES, "known_source.csv", (str(csv_path), None))

    store = VectorStoreService(persist_dir=str(tmp_path / "chroma2"), collection_name="known_source_test")
    store.add_documents(
        [Document(page_content="embedded chunk text, not what should be shown", metadata={
            "ticket_id": "MESOS-1", "summary": "Docker daemon crash", "component": "docker",
            "status": "Resolved", "chunk_index": 0, "source_file": "known_source.csv",
        })],
        FakeEmbeddingService(),
    )
    import app.vectorstore.chroma_store as chroma_store
    monkeypatch.setattr(chroma_store, "VectorStoreService", lambda: store)

    at = _page(ticket="MESOS-1")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.text)
    assert "The daemon crashes on boot" in text
    assert "Restart the daemon service" in text
    assert "embedded chunk text, not what should be shown" not in text  # CSV text used, not the chunk
    assert not any("Chunk 1 of" in e.label for e in at.expander)  # no chunk fallback UI shown
    # comments rendered as separate entries with formatted (not raw ISO) timestamps
    captions = [c.value for c in at.caption]
    assert any("Jan 01, 2020" in c for c in captions)
    assert any("Jan 02, 2020" in c for c in captions)
    assert "2020-01-01T00:00:00.000+0000" not in text  # raw timestamp isn't shown verbatim


def test_metadata_grid_shows_human_formatted_dates_not_raw_iso_timestamps():
    # st.badge isn't deeply inspectable via AppTest (renders as an opaque element,
    # like st.page_link), so this checks the part that IS verifiable: dates.
    at = _page(ticket="MESOS-1")
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "2020-01-05T00:00:00.000+0000" not in text
    assert "Jan 05, 2020" in text  # resolved_date, human formatted
