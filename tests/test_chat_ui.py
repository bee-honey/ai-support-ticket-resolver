"""Chat page and app shell, driven headlessly with AppTest against a fake service (no API calls)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.models.schemas import RAGResult, RAGSource, RetrievedChunk
from app.rag.service import StreamEvent
from evals.schemas import EvalQuery, load_labels, load_traces, write_jsonl

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
        chunk = RetrievedChunk(text="Docker fails to start", metadata={"ticket_id": "MESOS-1", "chunk_index": 0})
        yield StreamEvent(
            "done",
            result=RAGResult(answer=ANSWER, sources=[source], chunks=[chunk], retrieval_seconds=0.4,
                             generation_seconds=1.1, input_tokens=900, output_tokens=40),
        )


@pytest.fixture(autouse=True)
def fakes(monkeypatch, tmp_path):
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
    # Chat logging/feedback (ui/views/chat.py) write under EVALS_DIR -- point it at a
    # temp folder so these tests never touch the real evals/ directory.
    monkeypatch.setenv("EVALS_DIR", str(tmp_path))
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


# ---- chat logging + feedback ---------------------------------------------------------------------


def test_answering_logs_a_chat_trace(tmp_path):
    at = _chat()
    at.chat_input[0].set_value("docker won't start").run()

    traces = load_traces(tmp_path / "chat_logs/log.jsonl")
    assert len(traces) == 1
    assert traces[0].question == "docker won't start"
    assert traces[0].answer == ANSWER
    assert traces[0].retrieved_ticket_ids == ["MESOS-1"]
    assert traces[0].run_id == "chat"


def test_a_feedback_widget_is_shown_under_the_answer():
    at = _chat()
    at.chat_input[0].set_value("q").run()
    assert len(at.feedback) == 1


def test_thumbs_up_saves_a_pass_label():
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(1).run()

    labels = load_labels(_labels_path(at))
    assert list(labels.values())[0].verdict == "pass"


def test_thumbs_down_shows_a_reason_box_and_saves_a_fail_label():
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(0).run()

    assert len(at.text_input) == 1  # reason box only appears after a thumbs-down
    labels = load_labels(_labels_path(at))
    label = list(labels.values())[0]
    assert label.verdict == "fail" and label.reason == ""

    at.text_input[0].set_value("cited the wrong ticket")  # queued -- forms don't commit until submitted
    _button(at, "Submit").click().run()
    label = list(load_labels(_labels_path(at)).values())[0]
    assert label.reason == "cited the wrong ticket"


def test_a_confirmation_is_shown_immediately_for_thumbs_up():
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(1).run()
    assert any("Thanks for the feedback" in c.value for c in at.caption)


def test_the_reason_boxs_own_first_render_does_not_look_like_a_submission():
    # Regression test: right after clicking thumbs-down, the reason box appearing
    # (untouched, value="") must not look like feedback was auto-submitted --
    # that's exactly the bug this form wrapping fixes. No "Thanks" until the form
    # is actually submitted; an inviting, non-final caption shows up to that point.
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(0).run()

    assert not any("Thanks for the feedback" in c.value for c in at.caption)
    assert any("add a reason" in c.value for c in at.caption)

    at.text_input[0].set_value("cited the wrong ticket")
    _button(at, "Submit").click().run()
    assert any("Thanks for the feedback" in c.value for c in at.caption)


def test_the_confirmation_stays_correct_across_an_unrelated_rerun():
    # A later, unrelated rerun (e.g. changing a filter) must not re-show the
    # "add a reason" prompt for a turn whose reason was already submitted, and
    # must not silently blank out an already-saved reason either.
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(0).run()
    at.text_input[0].set_value("cited the wrong ticket")
    _button(at, "Submit").click().run()

    at.multiselect[0].select("docker").run()  # unrelated rerun
    assert any("Thanks for the feedback" in c.value for c in at.caption)
    label = list(load_labels(_labels_path(at)).values())[0]
    assert label.reason == "cited the wrong ticket"  # not clobbered back to empty


def test_rerunning_without_changing_feedback_does_not_duplicate_the_label(tmp_path):
    at = _chat()
    at.chat_input[0].set_value("q").run()
    at.feedback[0].set_value(1).run()
    at.multiselect[0].select("docker").run()  # an unrelated rerun

    lines = (tmp_path / "labels/chat_feedback.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def _labels_path(at: AppTest) -> Path:
    return Path(os.environ["EVALS_DIR"]) / "labels/chat_feedback.jsonl"


# ---- Recent Hot Issues ---------------------------------------------------------------------------


def _write_seed_dataset(tmp_path: Path, questions: list[str]) -> None:
    write_jsonl(
        tmp_path / "datasets/team_test_cases.jsonl",
        [EvalQuery(id=f"TC-{i:02d}", question=q, kind="answerable") for i, q in enumerate(questions, start=1)],
    )


def _hot_issues_popover(at: AppTest):
    # st.popover isn't exposed via a flat at.button/at.popover list -- it's a
    # container Block, found by the key= given to it (AppTest raises KeyError if
    # nothing in the tree used that key this run, e.g. render_hot_issues() returned
    # early with no suggestions to show). The key carries a generation suffix that
    # bumps every time a suggestion is clicked (see render_hot_issues), so look it
    # up rather than assuming generation 0.
    generation = at.session_state.get("_hot_issues_popover_generation", 0)
    try:
        return at.get_by_key(f"hot_issues_fab_{generation}")
    except KeyError:
        return None


def test_hot_issues_panel_shows_seed_suggestions_on_a_fresh_deployment(tmp_path):
    # No chat activity logged yet -- falls back entirely to the SME test set.
    _write_seed_dataset(tmp_path, ["docker fails to start", "network partition issue"])
    at = _chat()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "docker fails to start" in labels
    assert "network partition issue" in labels


def test_hot_issues_injected_html_never_shows_up_as_literal_text(tmp_path):
    # Regression test: a triple-quoted f-string indented to match the surrounding
    # Python code carries that same indentation into the actual string content --
    # and a line indented 4+ spaces is exactly what Markdown treats as a literal
    # code block, rendering raw injected HTML as visible escaped text on the page
    # instead of parsing it. textwrap.dedent() fixes it; this pins the fix by
    # asserting the block-starting lines of the injected markup start at column 0
    # (no leading whitespace at all, not just "less than 4 spaces").
    _write_seed_dataset(tmp_path, ["docker fails to start"])
    at = _chat()
    injected = next(m for m in at.markdown if "hot-issues-glow" in m.value)
    lines = [line for line in injected.value.splitlines() if line.strip()]
    # The property that actually matters: the tags that OPEN an HTML block --
    # nested CSS property lines inside an already-open <style> block are fine to
    # be indented, only the block-starting lines themselves must not be.
    assert lines[0] == "<style>"
    assert lines[-1] == "</style>"


def test_clicking_a_hot_issue_submits_it_as_a_question(tmp_path):
    _write_seed_dataset(tmp_path, ["docker fails to start"])
    at = _chat()
    _button(at, "docker fails to start").click().run()
    assert not at.exception
    assert FakeService.last_call[0] == "docker fails to start"
    assert len(at.session_state.history) == 1
    assert at.session_state.history[0]["question"] == "docker fails to start"


def test_hot_issues_button_remains_available_after_a_conversation_has_started(tmp_path):
    # Regression test for the reported bug: the panel used to only be shown before
    # the first question, so clicking a suggestion appeared to silently do nothing
    # on any later turn (it had already been hidden). It must stay available and
    # clickable for the whole session now, not just the first turn.
    _write_seed_dataset(tmp_path, ["docker fails to start"])
    at = _chat()
    at.chat_input[0].set_value("some other question").run()
    assert not at.exception
    assert len(at.session_state.history) == 1
    assert _hot_issues_popover(at) is not None  # the floating trigger is still there


def test_clicking_a_hot_issue_works_on_a_second_turn_too(tmp_path):
    _write_seed_dataset(tmp_path, ["docker fails to start"])
    at = _chat()
    at.chat_input[0].set_value("first question").run()
    assert len(at.session_state.history) == 1

    _button(at, "docker fails to start").click().run()
    assert not at.exception
    assert FakeService.last_call[0] == "docker fails to start"
    assert len(at.session_state.history) == 2
    assert at.session_state.history[1]["question"] == "docker fails to start"


def test_clicking_a_hot_issue_drops_it_and_backfills_from_the_rest_of_the_pool(tmp_path):
    _write_seed_dataset(tmp_path, ["docker fails to start", "network partition issue"])
    at = _chat()
    assert "docker fails to start" in [b.label for b in at.button]

    _button(at, "docker fails to start").click().run()
    # the click is recorded (added to the dismissed set) during this same run, but the
    # button list for THIS run was already decided before that -- the exclusion only
    # shows up on the next rerun, same as the existing feedback-state tests below.
    at.run()
    labels = [b.label for b in at.button]
    assert "docker fails to start" not in labels  # dropped, already asked this session
    assert "network partition issue" in labels  # the other one is still offered


def test_clicking_a_hot_issue_bumps_the_popover_generation_so_it_reopens_closed(tmp_path):
    # st.popover has no API to close it programmatically, and simply clicking
    # something inside it does not close it either -- the one lever that does
    # work is mounting a genuinely new widget instance (a new key), which starts
    # closed. This pins that the generation counter (and therefore the popover's
    # key) actually changes on a click, which is what makes that trick work.
    _write_seed_dataset(tmp_path, ["docker fails to start", "network partition issue"])
    at = _chat()
    assert at.session_state.get("_hot_issues_popover_generation", 0) == 0

    _button(at, "docker fails to start").click().run()
    assert at.session_state["_hot_issues_popover_generation"] == 1
    assert _hot_issues_popover(at) is not None  # still there (1 suggestion left), just a fresh instance


def test_hot_issues_button_does_not_appear_with_no_seed_dataset_and_no_chat_history():
    # No evals/datasets/team_test_cases.jsonl in this temp EVALS_DIR at all.
    at = _chat()
    assert not at.exception
    assert _hot_issues_popover(at) is None
