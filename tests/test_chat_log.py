"""log_chat_turn: every live chat answer persisted as a Trace, same shape eval runs use."""

from __future__ import annotations

from app.models.schemas import RAGResult, RAGSource, RetrievedChunk
from evals.chat_log import CHAT_RUN_ID, log_chat_turn
from evals.schemas import load_traces


def _result(**overrides):
    defaults = dict(
        answer="Suggested Resolution\n- restart docker\n\nSupporting Evidence\n- MESOS-1",
        sources=[RAGSource("MESOS-1", "docker", "Resolved", "Docker fails", "2023-01-14", None)],
        chunks=[RetrievedChunk(text="evidence", metadata={"ticket_id": "MESOS-1", "chunk_index": 0})],
        retrieval_seconds=0.3,
        generation_seconds=1.1,
        input_tokens=500,
        output_tokens=40,
    )
    defaults.update(overrides)
    return RAGResult(**defaults)


def test_log_chat_turn_writes_a_trace_with_a_stable_trace_id(tmp_path):
    log_path = tmp_path / "log.jsonl"
    trace = log_chat_turn("docker won't start", _result(), chat_model="gpt-4o-mini", top_k=5, log_path=log_path)

    assert trace.run_id == CHAT_RUN_ID
    assert trace.trace_id == f"{CHAT_RUN_ID}:{trace.query_id}"
    assert trace.question == "docker won't start"
    assert trace.answer.startswith("Suggested Resolution")
    assert trace.retrieved_ticket_ids == ["MESOS-1"]
    assert trace.chat_model == "gpt-4o-mini" and trace.top_k == 5
    assert round(trace.total_seconds, 4) == 1.4
    assert trace.checks == {} and trace.judgments == {}  # no ground truth to check/judge against


def test_log_chat_turn_persists_to_the_given_path(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_chat_turn("q1", _result(), chat_model="m", top_k=None, log_path=log_path)
    log_chat_turn("q2", _result(), chat_model="m", top_k=None, log_path=log_path)

    loaded = load_traces(log_path)
    assert [t.question for t in loaded] == ["q1", "q2"]


def test_log_chat_turn_handles_a_result_with_no_retrieved_chunks(tmp_path):
    log_path = tmp_path / "log.jsonl"
    trace = log_chat_turn(
        "unrelated question",
        _result(chunks=[], sources=[], answer="There is not enough supporting evidence..."),
        chat_model="m",
        top_k=5,
        log_path=log_path,
    )
    assert trace.retrieved_ticket_ids == [] and trace.context == ""


def test_the_logged_question_is_pii_redacted_even_if_the_caller_passes_the_raw_text():
    # RAGService.answer()/stream_answer() redact their OWN local copy of the question
    # internally -- that never reaches back to the caller's variable, so the chat UI
    # calls this with the raw text. This is the one place PII actually hits disk, so
    # it has to redact again itself rather than trust the caller already did.
    trace = log_chat_turn(
        "contact me at jane@example.com about this", _result(), chat_model="m", top_k=None, log_path="/dev/null"
    )
    assert "jane@example.com" not in trace.question
    assert "[redacted-email]" in trace.question


def test_each_call_gets_a_distinct_query_id():
    trace1 = log_chat_turn("q", _result(), chat_model="m", top_k=None, log_path="/dev/null")
    trace2 = log_chat_turn("q", _result(), chat_model="m", top_k=None, log_path="/dev/null")
    assert trace1.query_id != trace2.query_id
