"""Chat page: ask a support question, get a grounded answer with sources.

This page contains NO Chroma/OpenAI calls of its own -- it only talks to
`RAGService`. That boundary matters: in Phase 2, Streamlit can be pointed at
a FastAPI backend instead, without this file's structure changing.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root (ui/views/chat.py)

from app.config.settings import get_settings, model_choices  # noqa: E402
from app.models.schemas import RAGResult, RAGSource  # noqa: E402
from app.rag.service import RAGService  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from evals.chat_log import hot_issues, log_chat_turn  # noqa: E402
from evals.schemas import HumanLabel, append_jsonl, load_queries  # noqa: E402
from ui.formatting import format_live_stats, format_stats  # noqa: E402
from ui.ticket_links import ticket_page_link  # noqa: E402

# Same EVALS_DIR override the Evals page uses (see ui/views/evals.py) -- so a test can
# point both pages at one temp folder, and so the two stay in exact agreement about
# where chat activity lives without importing each other's page module.
EVALS_DIR = Path(os.getenv("EVALS_DIR") or Path(__file__).resolve().parent.parent.parent / "evals")
CHAT_LOG_PATH = EVALS_DIR / "chat_logs/log.jsonl"
CHAT_FEEDBACK_PATH = EVALS_DIR / "labels/chat_feedback.jsonl"
# Fallback source for the "Recent Hot Issues" panel until there's enough real chat
# traffic logged to be interesting (see get_suggested_questions below) -- real,
# SME-authored questions, not placeholder text.
SEED_SUGGESTIONS_DATASET = EVALS_DIR / "datasets/team_test_cases.jsonl"
SUGGESTION_COUNT = 5
# get_suggested_questions() fetches a bigger pool than SUGGESTION_COUNT so that
# render_hot_issues() can drop a question once it's been clicked this session and
# still backfill back up to SUGGESTION_COUNT from what's left, instead of the list
# just shrinking by one every click.
SUGGESTION_POOL_SIZE = SUGGESTION_COUNT * 3

# Fixed corner offset for the floating hot-issues button (see render_hot_issues).
# Not centered on any edge, so the ring's `scale(...)` animation never has to be
# combined with a positional `translate(...)`.
# Single line, deliberately -- an embedded newline here (even one with its own,
# different indentation) breaks textwrap.dedent()'s common-leading-whitespace
# calculation across the whole template it gets substituted into (see
# render_hot_issues), silently leaving every line indented and re-triggering the
# exact "shows up as literal text" bug dedent exists to prevent.
HOT_ISSUES_POSITION_CSS = "bottom: 140px; right: 80px;"

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


def _seed_suggestions() -> list[str]:
    """A few real, SME-authored answerable questions from the eval test set --
    used to top up get_suggested_questions() when there isn't enough real chat
    traffic logged yet, so the panel is never empty on a fresh deployment."""
    try:
        queries = load_queries(SEED_SUGGESTIONS_DATASET)
    except Exception:
        return []
    return [q.question for q in queries if q.kind == "answerable"]


@st.cache_data(ttl=60)
def get_suggested_questions() -> list[str]:
    """Up to SUGGESTION_POOL_SIZE "Recent Hot Issues" candidates: real, most-asked
    questions from live chat traffic (evals.chat_log.hot_issues), topped up with
    SME-authored example questions when there isn't enough real traffic yet -- so
    the panel starts useful on a fresh deployment and becomes more "real" as usage
    grows. A short ttl (not the 300s the metadata caches above use) since this is
    meant to feel current, not a slowly-changing reference list.

    Deliberately a bigger pool than SUGGESTION_COUNT (what's actually shown at
    once) -- see render_hot_issues, which drops a question from what it displays
    once it's been clicked this session and backfills from the rest of this pool,
    shared (and cached) across sessions; only which ones a given session has
    already dismissed is per-session.
    """
    try:
        suggestions = hot_issues(CHAT_LOG_PATH, CHAT_FEEDBACK_PATH, limit=SUGGESTION_POOL_SIZE)
    except Exception:
        suggestions = []
    seen = {q.strip().lower() for q in suggestions}
    for question in _seed_suggestions():
        if len(suggestions) >= SUGGESTION_POOL_SIZE:
            break
        key = question.strip().lower()
        if key not in seen:
            suggestions.append(question)
            seen.add(key)
    return suggestions


def render_hot_issues() -> str | None:
    """A floating 🔥 button (always available -- not just before the first
    question, unlike the earlier version of this panel) that opens a small
    popover of clickable "Recent Hot Issues" suggestions. Returns the clicked
    question's text this run (to be treated exactly like a typed-and-submitted
    question), or None if nothing was clicked.

    Click-to-open, not hover-to-open: `st.popover`'s content is mounted/unmounted
    via Streamlit's own internal state on click, not shown/hidden by CSS, so a
    pure-CSS `:hover` override can't force it open -- and hover has no equivalent
    on touch devices anyway, so click is the more robust choice, not just the
    easier one. `st.chat_input` also has no API to pre-fill it with text, so
    clicking a suggestion submits it directly (generates a response immediately)
    rather than "inserting then waiting for Enter", which isn't achievable here.

    A clicked suggestion is dropped from what THIS session sees again (backfilled
    from get_suggested_questions()'s bigger shared pool) -- asking a question
    once means you don't need it re-suggested, but it's still a legitimate
    suggestion for anyone else's session. Clicking one also shrinks the button
    (and drops the attention-grabbing glow ring) for the rest of the session --
    once it's actually been used, staying large/pulsing would just compete with
    the answer for attention; it's still fully clickable at the smaller size to
    reopen the popover and use it again.

    Position is a fixed corner offset (HOT_ISSUES_POSITION_CSS), not user-picked
    or draggable: Streamlit's `unsafe_allow_html` is documented to not reliably
    execute injected `<script>` tags (a deliberate security choice), so real
    click-and-drag would need a full custom Streamlit component (its own JS
    build, a Python<->JS bridge) -- disproportionate engineering for a placement
    preference, so one fixed spot it is.

    The floating position/styling is CSS keyed to the popover's own `key=`
    (Streamlit's documented hook for custom styling -- generates a
    `st-key-<key>` class on its wrapper), not to any internal/undocumented DOM
    structure -- more likely to keep working across Streamlit versions, but
    still worth eyeballing locally, since exact pixel placement can shift with
    the app's own layout. The glow is a separate fixed-position ring element
    (not baked into the button's own background/box-shadow), sized and
    positioned identically to the button and animated with its own `scale()` --
    since it shares the button's exact center, growing it outward reads as an
    expanding halo around the button rather than a static box-shadow.

    The injected HTML/CSS is run through `textwrap.dedent` before being handed
    to `st.markdown` -- without it, a plain triple-quoted string here carries
    the same indentation as the surrounding Python code, and a line indented
    4+ spaces is exactly what Markdown treats as a literal code block: the
    `<div>` rendered as visible escaped text on the page instead of being
    parsed as HTML (a real bug this fixes, not just a style nit).
    """
    dismissed = st.session_state.setdefault("_dismissed_hot_issues", set())
    suggestions = [q for q in get_suggested_questions() if q.strip().lower() not in dismissed][:SUGGESTION_COUNT]
    if not suggestions:
        return None
    minimized = st.session_state.get("_hot_issues_minimized", False)
    size = 30 if minimized else 52
    font_size = 14 if minimized else 22
    st.markdown(
        textwrap.dedent(f"""
        <style>
        @keyframes hot-issues-ring {{
            0%   {{ transform: scale(1);   opacity: 0.4; }}
            100% {{ transform: scale(2.2); opacity: 0; }}
        }}
        .hot-issues-ring {{
            position: fixed;
            {HOT_ISSUES_POSITION_CSS}
            width: {size}px;
            height: {size}px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 122, 61, 0.5) 0%, rgba(255, 122, 61, 0) 72%);
            animation: {"none" if minimized else "hot-issues-ring 1.8s ease-out infinite"};
            z-index: 998;
            pointer-events: none;
        }}
        div[class*="st-key-hot_issues_fab"] {{
            position: fixed;
            {HOT_ISSUES_POSITION_CSS}
            z-index: 999;
        }}
        div[class*="st-key-hot_issues_fab"] button {{
            border-radius: 50%;
            width: {size}px;
            height: {size}px;
            font-size: {font_size}px;
            border: none;
            background: linear-gradient(135deg, #ff9142, #ff5f6d);
            color: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        div[class*="st-key-hot_issues_fab"] button:hover {{
            filter: brightness(1.08);
        }}
        </style>
        <div class="hot-issues-ring"></div>
        """),
        unsafe_allow_html=True,
    )
    clicked = None
    with st.popover("🔥", key="hot_issues_fab", help="Recent Hot Issues"):
        st.markdown("**🔥 Recent Hot Issues**")
        for index, question in enumerate(suggestions):
            label = question if len(question) <= 80 else question[:77] + "..."
            if st.button(label, key=f"hot_issue_{index}", use_container_width=True):
                clicked = question
                dismissed.add(question.strip().lower())
                st.session_state["_hot_issues_minimized"] = True
    return clicked


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

# Always available, not just before the first question (see render_hot_issues).
suggested_question = render_hot_issues()

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

question = st.chat_input("Describe the support problem...") or suggested_question

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
