"""Deterministic eval checks: retrieval hit, citation validity, abstention, format, latency."""

from __future__ import annotations

from app.rag.service import NO_EVIDENCE_ANSWER
from evals.checks import cited_ticket_ids, is_abstention, run_checks
from evals.schemas import Trace

GOOD_ANSWER = "Suggested Resolution\n- restart the agent\n\nSupporting Evidence\n- MESOS-1 and MESOS-2"


def _trace(**overrides) -> Trace:
    base = dict(
        run_id="r", query_id="q1", question="q", kind="answerable", chat_model="m", top_k=5,
        expected_ticket_id="MESOS-1", answer=GOOD_ANSWER, retrieved_ticket_ids=["MESOS-1", "MESOS-2"],
        total_seconds=2.0,
    )
    base.update(overrides)
    return Trace(**base)


def test_all_checks_pass_for_a_good_answerable_trace():
    assert run_checks(_trace()) == {
        "retrieval_hit": True, "citations_valid": True, "abstention_correct": True,
        "has_required_sections": True, "within_latency": True,
    }


def test_retrieval_hit_fails_when_expected_ticket_missing_and_is_na_without_expected():
    assert run_checks(_trace(retrieved_ticket_ids=["MESOS-9"]))["retrieval_hit"] is False
    assert run_checks(_trace(expected_ticket_id=None))["retrieval_hit"] is None


def test_invented_citation_is_caught():
    checks = run_checks(_trace(answer=GOOD_ANSWER + " also MESOS-999"))
    assert checks["citations_valid"] is False


def test_abstention_is_correct_only_for_unanswerable_queries():
    abstain = _trace(kind="unanswerable", expected_ticket_id=None, answer=NO_EVIDENCE_ANSWER)
    assert run_checks(abstain)["abstention_correct"] is True
    assert run_checks(_trace(kind="unanswerable", expected_ticket_id=None))["abstention_correct"] is False
    # abstaining on an answerable query is a failure (false abstention)
    assert run_checks(_trace(answer=NO_EVIDENCE_ANSWER))["abstention_correct"] is False


def test_llm_phrased_abstention_is_recognised():
    assert is_abstention("There is not enough supporting evidence to recommend a resolution.")
    assert not is_abstention(GOOD_ANSWER)


def test_required_sections_skipped_for_abstentions_and_flagged_when_missing():
    assert run_checks(_trace(answer="just restart it MESOS-1"))["has_required_sections"] is False
    assert run_checks(_trace(answer=NO_EVIDENCE_ANSWER))["has_required_sections"] is None


def test_latency_cap():
    assert run_checks(_trace(total_seconds=12.0))["within_latency"] is False
    assert run_checks(_trace(total_seconds=12.0), latency_cap_seconds=15)["within_latency"] is True


def test_errored_trace_yields_no_answer_level_verdicts():
    checks = run_checks(_trace(error="boom", answer=""))
    assert checks["citations_valid"] is None and checks["within_latency"] is None
    assert checks["abstention_correct"] is None


def test_cited_ticket_ids_extracts_project_style_ids():
    assert cited_ticket_ids("see MESOS-12, and SPARK-3; not mesos-4 or ABC") == {"MESOS-12", "SPARK-3"}
