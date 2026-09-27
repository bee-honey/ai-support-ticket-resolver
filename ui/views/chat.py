"""Chat page: ask a support question, get a grounded answer with sources.

This page contains NO Chroma/OpenAI calls of its own -- it only talks to
`RAGService`. That boundary matters: in Phase 2, Streamlit can be pointed at
a FastAPI backend instead, without this file's structure changing.
"""

from __future__ import annotations

import os
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
from evals.chat_log import log_chat_turn  # noqa: E402
from evals.schemas import HumanLabel, append_jsonl  # noqa: E402
from ui.formatting import format_live_stats, format_stats  # noqa: E402
from ui.ticket_links import ticket_page_link  # noqa: E402

# Same EVALS_DIR override the Evals page uses (see ui/views/evals.py) -- so a test can
# point both pages at one temp folder, and so the two stay in exact agreement about
# where chat activity lives without importing each other's page module.
EVALS_DIR = Path(os.getenv("EVALS_DIR") or Path(__file__).resolve().parent.parent.parent / "evals")
CHAT_LOG_PATH = EVALS_DIR / "chat_logs/log.jsonl"
CHAT_FEEDBACK_PATH = EVALS_DIR / "labels/chat_feedback.jsonl"

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


def _save_feedback(trace_id: str, sentiment: int, reason: str, saved: dict[str, tuple[int, str]]) -> None:
    current = (sentiment, reason.strip())
    if saved.get(trace_id) == current:
        return
    try:
        append_jsonl(
            CHAT_FEEDBACK_PATH,
            HumanLabel(trace_id=trace_id, verdict="pass" if sentiment == 1 else "fail", reason=reason.strip()),
        )
    except Exception:
        return  # feedback is a bonus signal -- a write failure shouldn't surface as a chat error
    saved[trace_id] = current


def render_feedback(trace_id: str) -> None:
    """Thumbs up/down on one answer -> a `HumanLabel` keyed to `trace_id`, the same
    schema the eval framework's SME labels use (see evals/label.py) -- but written to
    its own file (`CHAT_FEEDBACK_PATH`), since this is raw end-user signal from live
    usage, not the calibration-grade ground truth `evals/align.py` compares judges
    against. Kept out of that comparison rather than silently mixed into it.

    Thumbs up saves and confirms immediately -- there's nothing more to ask. Thumbs
    down saves the bare verdict right away too (a real signal even if no reason ever
    follows), but the optional reason box lives inside a form: a bare `st.text_input`
    reruns the script (and, before this fix, saved + confirmed) on its own very
    first, untouched render, which looked exactly like feedback being auto-submitted
    before there was any chance to type something. A form only commits its
    contents on an explicit Submit (or Enter while focused inside it), so simply
    seeing the box appear no longer looks like a submission.
    """
    sentiment = st.feedback("thumbs", key=f"fb_{trace_id}")
    if sentiment is None:
        return

    saved = st.session_state.setdefault("_feedback_saved", {})

    if sentiment == 1:
        _save_feedback(trace_id, sentiment, "", saved)
        if trace_id in saved:
            st.caption("✅ Thanks for the feedback!")
        return

    if trace_id not in saved:  # first time seeing this thumbs-down: save the bare verdict once
        _save_feedback(trace_id, sentiment, "", saved)

    with st.form(key=f"fb_form_{trace_id}", border=False):
        reason = st.text_input(
            "What was wrong?", placeholder="What was wrong? (optional)", label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Submit")
    if submitted:
        _save_feedback(trace_id, sentiment, reason, saved)

    current = saved.get(trace_id)
    if current and current[1]:
        st.caption("✅ Thanks for the feedback!")
    elif current is not None:
        st.caption("Got it — add a reason above if you'd like.")


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
        if turn.get("trace_id"):
            render_feedback(turn["trace_id"])

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
            try:
                trace = log_chat_turn(question, result, chat_model=chat_model, top_k=top_k, log_path=CHAT_LOG_PATH)
            except Exception:
                trace = None  # logging is a bonus signal -- don't let it break a working answer
            if trace is not None:
                render_feedback(trace.trace_id)
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "sources": result.sources,
                    "result": result,
                    "model": chat_model,
                    "trace_id": trace.trace_id if trace is not None else None,
                }
            )
