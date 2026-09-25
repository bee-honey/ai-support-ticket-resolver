"""Tickets page: browse the ingested corpus, verify a citation by eye.

Read-only. This is the same data the chatbot cites and the Evals framework
measures against -- letting a ticket be opened directly means a citation
never has to be trusted blind. Deep-linkable via `?ticket=<id>` (used by the
chat page's Sources list and the Evals trace inspector).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root (ui/views/tickets.py)

from app.models.schemas import RetrievedChunk  # noqa: E402
from app.vectorstore.chroma_store import VectorStoreService  # noqa: E402

ANY_OPTION = "(any)"


@st.cache_resource
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@st.cache_data(ttl=300)
def load_ticket_index(_vector_store: VectorStoreService) -> pd.DataFrame:
    """One row per ticket (its first chunk's metadata) -- the browsable list."""
    columns = ["ticket_id", "summary", "component", "status", "issue_type", "resolved_date", "source_file"]
    try:
        chunks = _vector_store.get_chunks(filters={"chunk_index": 0})
    except Exception:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "ticket_id": c.metadata["ticket_id"],
            "summary": c.metadata.get("summary", ""),
            "component": (c.metadata.get("component") or "").replace(";", ", "),
            "status": c.metadata.get("status", ""),
            "issue_type": c.metadata.get("issue_type", ""),
            "resolved_date": (c.metadata.get("resolved_date") or "")[:10],  # date only, drop the time
            "source_file": c.metadata.get("source_file", ""),
        }
        for c in chunks
        if c.metadata.get("ticket_id")  # non-ticket docs (e.g. a runbook PDF) have no ticket_id; not browsable here
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("ticket_id", kind="stable").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_ticket_chunks(_vector_store: VectorStoreService, ticket_id: str) -> list[RetrievedChunk]:
    """All chunks for one ticket, in the order they were embedded."""
    chunks = _vector_store.get_chunks(filters={"ticket_id": ticket_id})
    return sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0))


def render_ticket_detail(vector_store: VectorStoreService, ticket_id: str) -> None:
    chunks = load_ticket_chunks(vector_store, ticket_id)
    if not chunks:
        st.error(f"No ticket found with id `{ticket_id}`.")
        return

    meta = chunks[0].metadata
    st.markdown(f"### {ticket_id}")
    if meta.get("summary"):
        st.markdown(f"**{meta['summary']}**")
    details = " · ".join(
        f"{label}: {value}"
        for label, value in (
            ("component", (meta.get("component") or "").replace(";", ", ")),
            ("status", meta.get("status")),
            ("type", meta.get("issue_type")),
            ("created", meta.get("created_date")),
            ("resolved", meta.get("resolved_date")),
            ("source", meta.get("source_file")),
        )
        if value
    )
    if details:
        st.caption(details)

    st.divider()
    st.caption(
        f"{len(chunks)} chunk(s) -- the actual units the retriever works with; a real query "
        "only ever returns some of these, never the whole ticket at once."
    )
    for chunk in chunks:
        index = chunk.metadata.get("chunk_index", 0)
        with st.expander(f"Chunk {index + 1} of {len(chunks)}", expanded=len(chunks) <= 3):
            st.text(chunk.text)


st.caption("Browse the tickets the chatbot is grounded in. Click a ticket to see exactly what it retrieves from.")

vector_store = get_vector_store()
index = load_ticket_index(vector_store)

if index.empty:
    st.info("No tickets are ingested yet. Ask a question on the Resolver page first -- it ingests data on first use.")
else:
    deep_linked_id = st.query_params.get("ticket")
    if deep_linked_id:
        render_ticket_detail(vector_store, deep_linked_id)
        if st.button("← Back to all tickets"):
            st.query_params.pop("ticket", None)
            st.rerun()
        st.divider()

    with st.sidebar:
        st.header("Filters")
        search = st.text_input("Search (id or summary)")
        status_options = [ANY_OPTION, *sorted(v for v in index["status"].unique() if v)]
        status_filter = st.selectbox("Status", status_options)
        component_tag_index = vector_store.component_tag_index()
        component_tags = st.multiselect("Component", sorted(component_tag_index))

    visible = index
    if search:
        needle = search.lower()
        visible = visible[
            visible["ticket_id"].str.lower().str.contains(needle) | visible["summary"].str.lower().str.contains(needle)
        ]
    if status_filter != ANY_OPTION:
        visible = visible[visible["status"] == status_filter]
    if component_tags:
        raw_matches = {raw for tag in component_tags for raw in component_tag_index.get(tag, [])}
        raw_display = {r.replace(";", ", ") for r in raw_matches}
        visible = visible[visible["component"].isin(raw_display)]

    st.caption(f"{len(visible)} of {len(index)} tickets")
    event = st.dataframe(
        visible,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="ticket_table",
    )
    selected_rows = event.selection.rows
    if selected_rows:
        chosen_id = visible.iloc[selected_rows[0]]["ticket_id"]
        if chosen_id != deep_linked_id:  # avoid re-rendering the same detail twice
            st.divider()
            render_ticket_detail(vector_store, chosen_id)
