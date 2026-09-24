"""Eval runner + report: trace capture, error handling, percentiles, pass-rate tables."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from app.models.schemas import RAGResult, RetrievedChunk
from evals.judges.base import METRICS, Judge
from evals.report import percentile, render_report
from evals.run import run_eval
from evals.schemas import EvalQuery, load_traces, write_jsonl

GOOD = "Suggested Resolution\n- fix\n\nSupporting Evidence\n- MESOS-1"


def _service(answers: dict[str, object]):
    """answers: question -> RAGResult | Exception."""
    service = MagicMock()

    def answer(question, k=None, filters=None):
        outcome = answers[question]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    service.answer.side_effect = answer
    return service


def _result(answer=GOOD, ticket="MESOS-1", retrieval=0.2, generation=1.0):
    return RAGResult(
        answer=answer, sources=[], chunks=[RetrievedChunk(text="evidence", metadata={"ticket_id": ticket, "chunk_index": 0})],
        retrieval_seconds=retrieval, generation_seconds=generation, input_tokens=100, output_tokens=20,
    )


def test_run_eval_captures_trace_fields_checks_and_errors():
    queries = [
        EvalQuery(id="q1", question="a", expected_ticket_ids=["MESOS-1"]),
        EvalQuery(id="q2", question="b", kind="unanswerable"),
        EvalQuery(id="q3", question="c"),
    ]
    service = _service({"a": _result(), "b": _result(answer=GOOD), "c": RuntimeError("openai down")})
    traces = run_eval(queries, service, chat_model="gpt-x", k=3, run_id="run1")

    first = traces[0]
    assert (first.run_id, first.chat_model, first.top_k, first.trace_id) == ("run1", "gpt-x", 3, "run1:q1")
    assert first.retrieved_ticket_ids == ["MESOS-1"] and "[Evidence 1]" in first.context
    assert first.total_seconds == 1.2 and first.total_tokens == 120
    assert first.checks["retrieval_hit"] is True
    assert traces[1].checks["abstention_correct"] is False  # answered an unanswerable query
    assert traces[2].error == "RuntimeError: openai down" and traces[2].answer == ""


def test_run_eval_passes_filters_and_k_to_the_service():
    service = _service({"a": _result()})
    run_eval([EvalQuery(id="q1", question="a", filters={"component": "docker"})], service, chat_model="m", k=7)
    service.answer.assert_called_once_with("a", k=7, filters={"component": "docker"})


def test_run_eval_with_judges_and_jsonl_roundtrip(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='{"score": 1, "reason": "fine"}')
    judges = {m: Judge(m, model="m", llm=llm) for m in METRICS}
    traces = run_eval([EvalQuery(id="q1", question="a")], _service({"a": _result()}), chat_model="m", judges=judges)
    path = tmp_path / "run.jsonl"
    write_jsonl(path, traces)
    loaded = load_traces(path)
    assert loaded[0].judgments["faithfulness"].score == 1
    assert json.loads(path.read_text())["query_id"] == "q1"


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(i) for i in range(1, 11)]
    assert percentile(values, 50) == 5.0 and percentile(values, 95) == 10.0


def test_report_shows_pass_rates_latency_and_failure_reasons():
    llm_ok = MagicMock(content='{"score": 1, "reason": "ok"}')
    llm_bad = MagicMock(content='{"score": 0, "reason": "invented a flag"}')
    queries = [EvalQuery(id="q1", question="a"), EvalQuery(id="q2", question="b")]
    traces = run_eval(queries, _service({"a": _result(), "b": _result()}), chat_model="m", run_id="r")
    faith = MagicMock()
    faith.invoke.side_effect = [llm_ok, llm_bad]
    from evals.judges.base import judge_traces
    judge_traces(traces, {"faithfulness": Judge("faithfulness", model="m", llm=faith)}, workers=1)

    report = render_report(traces)
    assert "faithfulness" in report and "1/2 (50%)" in report
    assert "invented a flag" in report
    assert "p50" in report and "tool calls n/a" in report
