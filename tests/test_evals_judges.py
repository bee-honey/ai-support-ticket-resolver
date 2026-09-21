"""LLM judges: verdict parsing, prompt inputs, short-circuits, error handling -- LLM mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evals.judges.base import METRICS, Judge, build_user_message, judge_traces, parse_verdict
from evals.schemas import Trace


def _trace(**overrides) -> Trace:
    base = dict(
        run_id="r", query_id="q1", question="Docker executor fails", kind="answerable", chat_model="m", top_k=5,
        answer="Restart docker. See MESOS-1.", context="[Evidence 1] ticket_id=MESOS-1\nrestart the daemon",
    )
    base.update(overrides)
    return Trace(**base)


def _llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


@pytest.mark.parametrize(
    "raw, score",
    [('{"score": 1, "reason": "ok"}', 1), ('{"score": "0", "reason": "bad"}', 0),
     ('```json\n{"score": 1, "reason": "fenced"}\n```', 1), ('Sure! {"score": 0, "reason": "x"} done', 0)],
)
def test_parse_verdict_accepts_common_shapes(raw, score):
    assert parse_verdict(raw).score == score


@pytest.mark.parametrize("raw", ["not json", '{"score": 7, "reason": "x"}', '{"reason": "no score"}', "[1]"])
def test_parse_verdict_returns_none_score_for_garbage(raw):
    assert parse_verdict(raw).score is None


def test_each_metric_only_sees_its_own_inputs():
    trace = _trace()
    context_msg = build_user_message("context_relevance", trace)
    assert "RETRIEVED EVIDENCE" in context_msg and "ANSWER" not in context_msg
    answer_msg = build_user_message("answer_relevancy", trace)
    assert "ANSWER" in answer_msg and "RETRIEVED EVIDENCE" not in answer_msg
    faithful_msg = build_user_message("faithfulness", trace)
    assert "RETRIEVED EVIDENCE" in faithful_msg and "ANSWER" in faithful_msg


def test_judge_returns_parsed_verdict_and_sends_system_prompt():
    llm = _llm('{"score": 1, "reason": "on topic"}')
    verdict = Judge("answer_relevancy", model="m", llm=llm).judge(_trace())
    assert (verdict.score, verdict.reason) == (1, "on topic")
    system, human = llm.invoke.call_args[0][0]
    assert "answer_relevancy" not in system.content and "Score 1" in system.content
    assert "Docker executor fails" in human.content


def test_every_metric_has_a_prompt_file():
    for metric in METRICS:
        assert Judge(metric, model="m", llm=_llm("{}"))._system_prompt


def test_llm_exception_becomes_unscored_verdict_not_a_crash():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("rate limited")
    verdict = Judge("faithfulness", model="m", llm=llm).judge(_trace())
    assert verdict.score is None and "rate limited" in verdict.reason


def test_empty_context_is_decided_without_calling_the_llm():
    llm = _llm('{"score": 1, "reason": "x"}')
    empty = _trace(context="", answer="There is not enough supporting evidence to recommend a resolution.")
    assert Judge("context_relevance", model="m", llm=llm).judge(empty).score == 0
    assert Judge("faithfulness", model="m", llm=llm).judge(empty).score == 1
    llm.invoke.assert_not_called()


def test_judge_traces_fills_all_metrics_and_skips_errored_traces():
    traces = [_trace(query_id="a"), _trace(query_id="b", error="boom")]
    judges = {m: Judge(m, model="m", llm=_llm('{"score": 1, "reason": "ok"}')) for m in METRICS}
    judge_traces(traces, judges, workers=2)
    assert set(traces[0].judgments) == set(METRICS)
    assert traces[1].judgments == {}
