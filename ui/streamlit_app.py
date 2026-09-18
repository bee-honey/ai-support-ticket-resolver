"""Simple Streamlit chatbot for the Support Ticket Resolver.

This UI contains NO Chroma/OpenAI calls of its own -- it only talks to
`RAGService`. That boundary matters: in Phase 2, Streamlit can be pointed at
a FastAPI backend instead, without this file's structure changing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.models.schemas import RAGSource  # noqa: E402
from app.rag.service import RAGService  # noqa: E402

st.set_page_config(page_title="Support Ticket Resolver", page_icon="🎫")


@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


@st.cache_data(ttl=300)
def get_filter_options(_rag_service: RAGService, field: str) -> list[str]:
    # Chroma's `where` filter is an exact, case-sensitive match, so offering
    # the real stored values (rather than free text) avoids silent
    # zero-result filters from casing/typo mismatches (e.g. "resolved" vs
    # the stored "Resolved").
    try:
        return _rag_service.retriever.vector_store.list_metadata_values(field)
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_component_tag_index(_rag_service: RAGService) -> dict[str, list[str]]:
    """Maps a single clean component tag (e.g. "docker") to every raw stored
    `component` string that includes it (e.g. "docker;agent").

    Tickets can list multiple components as one semicolon-joined string
    (e.g. "agent;containerization;libprocess;stout"), which is unreadable as
    a flat dropdown of ~100 combinations. Splitting it into tags for display,
    then mapping a selected tag back to every raw string containing it, lets
    the UI offer clean choices while still filtering with Chroma's exact-match
    `where` (via `$in` over the matching raw strings).
    """
    try:
        raw_values = _rag_service.retriever.vector_store.list_metadata_values("component")
    except Exception:
        return {}

    index: dict[str, list[str]] = {}
    for raw in raw_values:
        for tag in (part.strip() for part in raw.split(";")):
            if tag:
                index.setdefault(tag, []).append(raw)
    return index


def render_source(source: RAGSource) -> str:
    title = source.ticket_id or source.source_file or "Unknown source"
    line1 = f"**{title}** — {source.summary}" if source.summary else f"**{title}**"
    component_display = source.component.replace(";", ", ") if source.component else None
    details = " · ".join(
        f"{label}: {value}"
        for label, value in (
            ("component", component_display),
            ("status", source.status),
            ("resolved", source.resolved_date),
        )
        if value
    )
    return f"{line1}\n\n{details}" if details else line1


st.title("🎫 Support Ticket Resolver")
st.caption("Ask about a support problem. Answers are grounded in historical tickets and documentation.")

settings = get_settings()
if not settings.openai_api_key or settings.openai_api_key == "sk-changeme":
    st.warning(
        "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add a real key before asking a question.",
        icon="⚠️",
    )

ANY_OPTION = "(any)"

with st.sidebar:
    st.header("Filters (optional)")
    rag_service = get_rag_service()
    component_tag_index = get_component_tag_index(rag_service)
    status_options = [ANY_OPTION, *get_filter_options(rag_service, "status")]
    component_tags = st.multiselect("Component", sorted(component_tag_index))
    status_filter = st.selectbox("Status", status_options)
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=settings.default_top_k)
    if st.button("Show collection info"):
        try:
            st.json(get_rag_service().retriever.vector_store.collection_info())
        except Exception as exc:  # e.g. collection not created yet
            st.error(f"Could not read collection info: {exc}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for source in turn["sources"]:
                    st.markdown(render_source(source))

question = st.chat_input("Describe the support problem...")

if question:
    filters: dict[str, Any] = {}
    if component_tags:
        raw_matches = sorted({raw for tag in component_tags for raw in component_tag_index.get(tag, [])})
        filters["component"] = {"$in": raw_matches}
    if status_filter != ANY_OPTION:
        filters["status"] = status_filter

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                result = get_rag_service().answer(question, k=top_k, filters=filters or None)
            except Exception as exc:
                st.error(f"Failed to generate an answer: {exc}")
                result = None

        if result is not None:
            st.write(result.answer)
            if result.sources:
                with st.expander(f"Sources ({len(result.sources)})"):
                    for source in result.sources:
                        st.markdown(render_source(source))
            st.session_state.history.append(
                {"question": question, "answer": result.answer, "sources": result.sources}
            )
