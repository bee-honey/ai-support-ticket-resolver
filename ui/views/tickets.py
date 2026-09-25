"""Tickets page: browse the ingested corpus, verify a citation by eye.

Read-only. This is the same data the chatbot cites and the Evals framework
measures against -- letting a ticket be opened directly means a citation
never has to be trusted blind. Deep-linkable via `?ticket=<id>` (used by the
chat page's Sources list and the Evals trace inspector).

The detail view shows the ticket's summary/description/comments/resolution
read fresh from its source CSV (app/ingestion/ticket_lookup.py) -- the whole
fields as they actually exist, not the embedded/chunked text split across
several overlapping pieces. If that lookup can't resolve (a non-CSV source,
or the file isn't reachable), it falls back to the retrieved-chunk view.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root (ui/views/tickets.py)

from app.ingestion.ticket_lookup import CommentEntry, TicketText, load_ticket_text  # noqa: E402
from app.models.schemas import RetrievedChunk  # noqa: E402
from app.vectorstore.chroma_store import VectorStoreService  # noqa: E402

ANY_OPTION = "(any)"
TABLE_HEIGHT = 420  # a fixed, moderate height keeps this a real "top half"; detail renders below it
LIST_COLUMNS = ["ticket_id", "summary", "component", "status", "resolved_date"]
LIST_HEADERS = ["ID", "Summary", "Component", "Status", "Resolved"]

STATUS_COLORS = {
    "resolved": "green", "closed": "green", "done": "green",
    "open": "blue", "new": "blue", "reopened": "blue",
    "in progress": "orange",
}


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
            "resolved_date": c.metadata.get("resolved_date", ""),
            "source_file": c.metadata.get("source_file", ""),
        }
        for c in chunks
        if c.metadata.get("ticket_id")  # non-ticket docs (e.g. a runbook PDF) have no ticket_id; not browsable here
    ]
    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("ticket_id", kind="stable").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_ticket_chunks(_vector_store: VectorStoreService, ticket_id: str) -> list[RetrievedChunk]:
    """All chunks for one ticket, in the order they were embedded -- the fallback view."""
    chunks = _vector_store.get_chunks(filters={"ticket_id": ticket_id})
    return sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0))


@st.cache_data(ttl=300)
def load_full_ticket_text(ticket_id: str, source_file: str) -> TicketText | None:
    return load_ticket_text(ticket_id, source_file)


def format_date(raw: str | None) -> str:
    """"2019-11-21T17:36:37.000+0000" -> "Nov 21, 2019" -- falls back to the raw
    value (or "") if it isn't parseable, rather than hiding a real value."""
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except ValueError:
        return raw


def format_datetime(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%b %d, %Y, %H:%M UTC")
    except ValueError:
        return raw


def render_metadata_grid(meta: dict[str, str | None]) -> None:
    """Labeled fields as a compact grid, dates formatted, status as a colored badge --
    the earlier one-line "component: x · status: y · ..." caption was hard to scan."""
    fields = [
        ("Component", (meta.get("component") or "").replace(";", ", ") or None),
        ("Status", meta.get("status")),
        ("Type", meta.get("issue_type")),
        ("Created", format_date(meta.get("created_date")) or None),
        ("Resolved", format_date(meta.get("resolved_date")) or None),
        ("Source", meta.get("source_file")),
    ]
    fields = [(label, value) for label, value in fields if value]
    if not fields:
        return
    columns = st.columns(min(len(fields), 3))
    for index, (label, value) in enumerate(fields):
        with columns[index % len(columns)]:
            st.caption(label)
            if label == "Status":
                st.badge(value, color=STATUS_COLORS.get(value.lower(), "gray"))
            else:
                st.markdown(f"**{value}**")


def render_comment(entry: CommentEntry) -> None:
    with st.container(border=True):
        if entry.timestamp:
            st.caption(format_datetime(entry.timestamp))
        st.write(entry.text)


def render_ticket_body_from_csv(text: TicketText) -> None:
    if text.description:
        st.markdown("**Problem**")
        st.write(text.description)
    if text.comments:
        st.markdown(f"**Comments** ({len(text.comments)})")
        for entry in text.comments:
            render_comment(entry)
    if text.resolution:
        st.markdown("**Resolution**")
        st.write(text.resolution)
    if not (text.description or text.comments or text.resolution):
        st.caption("This ticket has no description, comments, or resolution on file.")


def render_ticket_body_from_chunks(vector_store: VectorStoreService, ticket_id: str) -> None:
    chunks = load_ticket_chunks(vector_store, ticket_id)
    if not chunks:
        st.caption("No content found for this ticket.")
        return
    st.caption(
        f"Source text isn't available -- showing the {len(chunks)} retrieved chunk(s) instead, "
        "the units the retriever actually works with."
    )
    for chunk in chunks:
        index = chunk.metadata.get("chunk_index", 0)
        with st.expander(f"Chunk {index + 1} of {len(chunks)}", expanded=len(chunks) <= 3):
            st.text(chunk.text)


def render_ticket_detail(vector_store: VectorStoreService, ticket_id: str) -> None:
    chunks = load_ticket_chunks(vector_store, ticket_id)
    if not chunks:
        st.error(f"No ticket found with id `{ticket_id}`.")
        return

    meta = chunks[0].metadata
    st.markdown(f"### {ticket_id}")
    if meta.get("summary"):
        st.markdown(f"**{meta['summary']}**")
    render_metadata_grid(meta)
    st.divider()

    text = load_full_ticket_text(ticket_id, meta.get("source_file", ""))
    if text is not None:
        render_ticket_body_from_csv(text)
    else:
        render_ticket_body_from_chunks(vector_store, ticket_id)


def clear_selection() -> None:
    st.query_params.pop("ticket", None)
    st.session_state.pop("ticket_table", None)  # also drop the dataframe's own remembered selection


def enter_browse_mode() -> None:
    st.session_state["tickets_browsing"] = True


st.caption("Browse the tickets the chatbot is grounded in. Check a row to see exactly what it retrieves from.")

vector_store = get_vector_store()
index = load_ticket_index(vector_store)

if index.empty:
    st.info("No tickets are ingested yet. Ask a question on the Resolver page first -- it ingests data on first use.")
else:
    deep_linked_id = st.query_params.get("ticket")
    # A *new* external deep link (a citation clicked from Chat/Evals) always gets the focused,
    # detail-only view below -- even if this session had browsed the full table earlier --
    # since clicking a citation should show just that ticket, not the whole tab. Browsing mode
    # (table + detail below it) persists only while staying on the same ticket/table.
    if deep_linked_id != st.session_state.get("tickets_last_seen"):
        st.session_state["tickets_browsing"] = False
    st.session_state["tickets_last_seen"] = deep_linked_id

    if deep_linked_id and not st.session_state.get("tickets_browsing"):
        render_ticket_detail(vector_store, deep_linked_id)
        st.button("← Browse all tickets", on_click=enter_browse_mode)
    else:
        search = st.text_input("Search tickets", placeholder="Search by ID or summary...", label_visibility="collapsed")
        left, right = st.columns(2)
        with left:
            status_options = [ANY_OPTION, *sorted(v for v in index["status"].unique() if v)]
            status_filter = st.selectbox("Status", status_options)
        with right:
            component_tag_index = vector_store.component_tag_index()
            component_tags = st.multiselect("Component", sorted(component_tag_index))

        visible = index
        if search:
            needle = search.lower()
            visible = visible[
                visible["ticket_id"].str.lower().str.contains(needle)
                | visible["summary"].str.lower().str.contains(needle)
            ]
        if status_filter != ANY_OPTION:
            visible = visible[visible["status"] == status_filter]
        if component_tags:
            raw_matches = {raw for tag in component_tags for raw in component_tag_index.get(tag, [])}
            raw_display = {r.replace(";", ", ") for r in raw_matches}
            visible = visible[visible["component"].isin(raw_display)]

        st.caption(f"{len(visible)} of {len(index)} tickets")

        display = visible[LIST_COLUMNS].copy()
        display["resolved_date"] = display["resolved_date"].map(format_date)
        display.columns = LIST_HEADERS
        event = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=TABLE_HEIGHT,
            on_select="rerun",
            selection_mode="single-row",
            key="ticket_table",
        )

        # A fresh row click always wins; otherwise fall back to the deep link (e.g. just
        # arrived via "Browse all tickets" with that ticket still the one in the URL).
        selected_rows = event.selection.rows
        selected_id = visible.iloc[selected_rows[0]]["ticket_id"] if selected_rows else deep_linked_id

        if selected_id:
            if selected_rows:  # keep the URL in sync with a live click -- stays a shareable deep link
                st.query_params["ticket"] = selected_id
                st.session_state["tickets_last_seen"] = selected_id
            st.divider()
            header, clear = st.columns([6, 1])
            with header:
                st.markdown("**Ticket details**")
            with clear:
                st.button("✕ Clear", help="Close this ticket's detail", on_click=clear_selection)
            render_ticket_detail(vector_store, selected_id)
