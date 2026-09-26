"""Chat page: ask a support question, get a grounded answer with sources.

This page contains NO Chroma/OpenAI calls of its own -- it only talks to
`RAGService`. That boundary matters: in Phase 2, Streamlit can be pointed at
a FastAPI backend instead, without this file's structure changing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root (ui/views/chat.py)

from app.config.settings import get_settings, model_choices  # noqa: E402
from app.models.schemas import RAGResult, RAGSource  # noqa: E402
from app.rag.service import RAGService  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from ui.formatting import format_live_stats, format_stats  # noqa: E402
from ui.ticket_links import ticket_page_link  # noqa: E402

# Repaint the streaming answer/metrics at most this often (seconds); a repaint per token is needless churn.
PAINT_INTERVAL = 0.05

# No page-specific max-width here: use the app's normal wide layout, same as Tickets and Evals,
# rather than narrowing just this page (which previously made its header/content and the
# chat_input bar inconsistent widths against the other tabs, and against each other).


@st.cache_resource
def get_retriever() -> Retriever:
    # One shared retriever (and Chroma client); switching the chat model only
    # rebuilds the cheap LLM wrapper below, never the vector store connection.
    return Retriever()


@st.cache_resource
def get_rag_service(chat_model: str) -> RAGService:
    return RAGService(retriever=get_retriever(), chat_model=chat_model)


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
    try:
        return _rag_service.retriever.vector_store.component_tag_index()
    except Exception:
        return {}


def render_source(source: RAGSource) -> None:
    """One source: its ticket ID as a clickable link to the Tickets page (so the
    citation can be verified by eye instead of trusted blind), plus its details."""
    if source.ticket_id:
        label = f"{source.ticket_id} — {source.summary}" if source.summary else source.ticket_id
        ticket_page_link(source.ticket_id, label=label)
    else:
        title = source.source_file or "Unknown source"
        st.markdown(f"**{title}** — {source.summary}" if source.summary else f"**{title}**")
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
    if details:
        st.caption(details)


st.caption("Describe a support problem. Answers are grounded in historical tickets and documentation.")

settings = get_settings()
if not settings.openai_api_key or settings.openai_api_key == "sk-changeme":
    st.warning(
        "OPENAI_API_KEY is not set. Copy `.env.example` to `.env` and add a real key before asking a question.",
        icon="⚠️",
    )

ANY_OPTION = "(any)"


def _toggle_collection_info() -> None:
    st.session_state["show_collection_info"] = not st.session_state.get("show_collection_info", False)


with st.sidebar:
    st.header("Model")
    chat_models = model_choices(settings.chat_model)
    chat_model = st.selectbox("Chat model", chat_models, index=chat_models.index(settings.chat_model))
    rag_service = get_rag_service(chat_model)
    st.header("Filters (optional)")
    component_tag_index = get_component_tag_index(rag_service)
    status_options = [ANY_OPTION, *get_filter_options(rag_service, "status")]
    component_tags = st.multiselect("Component", sorted(component_tag_index))
    status_filter = st.selectbox("Status", status_options)
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=settings.default_top_k)
    showing_info = st.session_state.get("show_collection_info", False)
    st.button(
        "Hide collection info" if showing_info else "Show collection info",
        key="collection_info_button",
        on_click=_toggle_collection_info,
    )
    if showing_info:
        try:
            st.json(rag_service.retriever.vector_store.collection_info())
        except Exception as exc:  # e.g. collection not created yet
            st.error(f"Could not read collection info: {exc}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        st.caption(format_stats(turn["result"], turn.get("model")))
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for source in turn["sources"]:
                    render_source(source)

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
        answer_slot = st.empty()
        stats_slot = st.empty()  # live metrics while working, final snapshot when done
        stats_slot.caption(format_live_stats(chat_model, "retrieving"))

        result: RAGResult | None = None
        text, tokens_out, retrieval_seconds = "", 0, 0.0
        generation_started = last_paint = time.perf_counter()
        try:
            for event in rag_service.stream_answer(question, k=top_k, filters=filters or None):
                if event.kind == "retrieved":
                    retrieval_seconds = event.seconds
                    generation_started = time.perf_counter()
                elif event.kind == "token":
                    text += event.text
                    tokens_out += 1  # the API streams roughly one token per chunk
                else:
                    result = event.result
                    continue
                now = time.perf_counter()
                if event.kind == "retrieved" or now - last_paint >= PAINT_INTERVAL:
                    last_paint = now
                    if text:
                        answer_slot.markdown(text + "▌")
                    stats_slot.caption(
                        format_live_stats(
                            chat_model,
                            "generating",
                            retrieval_seconds=retrieval_seconds,
                            generation_seconds=now - generation_started,
                            tokens_out=tokens_out,
                        )
                    )
        except Exception as exc:
            answer_slot.empty()
            stats_slot.empty()
            st.error(f"Failed to generate an answer: {exc}")

        if result is not None:
            answer_slot.markdown(result.answer)
            stats_slot.caption(format_stats(result, chat_model))
            if result.sources:
                with st.expander(f"Sources ({len(result.sources)})"):
                    for source in result.sources:
                        render_source(source)
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "sources": result.sources,
                    "result": result,
                    "model": chat_model,
                }
            )
