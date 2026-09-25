"""Chat page and app shell, driven headlessly with AppTest against a fake service (no API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.models.schemas import RAGResult, RAGSource
from app.rag.service import StreamEvent

UI = Path(__file__).resolve().parent.parent / "ui"
APP = str(UI / "resolver_support.py")

ANSWER = "Suggested Resolution\n- restart docker"


class FakeVectorStore:
    def list_metadata_values(self, field):
        return {"component": ["docker", "agent;webui"], "status": ["Resolved"]}.get(field, [])

    def component_tag_index(self) -> dict[str, list[str]]:
        return {"docker": ["docker"], "agent": ["agent;webui"], "webui": ["agent;webui"]}

    def collection_info(self):
        return {"name": "support_tickets", "count": 3, "persist_dir": "chroma_db"}


class FakeService:
    fail = False

    def __init__(self, retriever=None, chat_model=None, **_):
        self.chat_model = chat_model
        self.retriever = MagicMock(vector_store=FakeVectorStore())

    def stream_answer(self, question, k=None, filters=None):
        FakeService.last_call = (question, k, filters)
        yield StreamEvent("retrieved", chunks=2, seconds=0.4)
        if FakeService.fail:
            raise RuntimeError("rate limited")
        yield StreamEvent("token", text="Suggested Resolution\n")
        yield StreamEvent("token", text="- restart docker")
        source = RAGSource("MESOS-1", "docker", "Resolved", "Docker fails", "2023-01-14", "t.csv")
        yield StreamEvent(
            "done",
            result=RAGResult(answer=ANSWER, sources=[source], retrieval_seconds=0.4, generation_seconds=1.1,
                             input_tokens=900, output_tokens=40),
        )


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    import app.rag.service as rag_service
    import app.retrieval.retriever as retriever
    import app.vectorstore.chroma_store as chroma_store

    FakeService.fail = False
    monkeypatch.setattr(rag_service, "RAGService", FakeService)
    monkeypatch.setattr(retriever, "Retriever", lambda: MagicMock())
    # The app shell's one-time ingest-if-empty bootstrap (ui/resolver_support.py)
    # builds its own VectorStoreService() directly -- patch it too, so it sees a
    # non-empty collection and never touches the real Chroma dir or OpenAI API
    # while these tests run through the full entrypoint below.
    monkeypatch.setattr(chroma_store, "VectorStoreService", FakeVectorStore)
    st.cache_resource.clear()
    st.cache_data.clear()


def _chat() -> AppTest:
    # Through the full entrypoint (not the bare page file): st.page_link (used by
    # ticket cross-links) needs an active st.navigation() context to resolve its
    # target, which only exists when the app is entered via resolver_support.py.
    # Chat is the default page, so this already lands there.
    return AppTest.from_file(APP, default_timeout=30).run()


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_collection_info_button_toggles_between_show_and_hide():
    at = _chat()
    assert not at.exception and len(at.json) == 0

    _button(at, "Show collection info").click().run()
    assert len(at.json) == 1 and "support_tickets" in at.json[0].value
    labels = [b.label for b in at.button]
    assert "Hide collection info" in labels and "Show collection info" not in labels

    _button(at, "Hide collection info").click().run()
    assert len(at.json) == 0
    assert "Show collection info" in [b.label for b in at.button]


def test_answer_streams_then_settles_on_final_metrics_and_sources():
    at = _chat()
    at.chat_input[0].set_value("docker won't start").run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assistant = at.chat_message[1]
    assert ANSWER in " ".join(m.value for m in assistant.markdown)
    caption = " ".join(c.value for c in assistant.caption)
    assert "🤖 gpt-4o-mini" in caption and "⏱ 1.5s" in caption and "940 tokens" in caption
    assert "▌" not in " ".join(m.value for m in assistant.markdown)  # streaming cursor is gone
    assert len(at.session_state.history) == 1
    assert at.session_state.history[0]["model"] == "gpt-4o-mini"


def test_previous_turns_keep_their_metrics_on_rerun():
    at = _chat()
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    captions = [c.value for m in at.chat_message for c in m.caption]
    assert sum("tokens" in c for c in captions) == 2


def test_filters_and_k_reach_the_service():
    at = _chat()
    at.multiselect[0].select("docker")
    at.slider[0].set_value(3)
    at.chat_input[0].set_value("q").run()
    question, k, filters = FakeService.last_call
    assert (question, k) == ("q", 3) and filters == {"component": {"$in": ["docker"]}}


def test_a_failing_stream_shows_an_error_and_no_partial_answer():
    FakeService.fail = True
    at = _chat()
    at.chat_input[0].set_value("q").run()
    assert any("Failed to generate an answer: rate limited" in e.value for e in at.error)
    assert at.session_state.history == []


def test_model_picker_switches_the_service_model():
    at = _chat()
    picker = next(s for s in at.selectbox if s.label == "Chat model")
    picker.select("gpt-4o").run()
    at.chat_input[0].set_value("q").run()
    assert at.session_state.history[0]["model"] == "gpt-4o"


def test_app_shell_renders_header_and_navigation():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    header = " ".join(m.value for m in at.markdown)
    assert "Resolver" in header and "Support" in header and "data:image/svg+xml;base64," in header
