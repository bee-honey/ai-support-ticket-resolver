"""Deterministic eval checks: retrieval hit, citation validity, abstention, format, latency."""

from __future__ import annotations

from app.rag.service import IRRELEVANT_EVIDENCE_ANSWER, NO_EVIDENCE_ANSWER
from evals.checks import cited_ticket_ids, has_required_sections, is_abstention, run_checks
from evals.schemas import Trace

GOOD_ANSWER = "Suggested Resolution\n- restart the agent\n\nSupporting Evidence\n- MESOS-1 and MESOS-2"


def _trace(**overrides) -> Trace:
    base = dict(
        run_id="r", query_id="q1", question="q", kind="answerable", chat_model="m", top_k=5,
        expected_ticket_ids=["MESOS-1"], answer=GOOD_ANSWER, retrieved_ticket_ids=["MESOS-1", "MESOS-2"],
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
    assert run_checks(_trace(expected_ticket_ids=[]))["retrieval_hit"] is None


def test_retrieval_hit_passes_if_any_one_of_several_expected_tickets_was_retrieved():
    checks = run_checks(_trace(expected_ticket_ids=["MESOS-404", "MESOS-2"], retrieved_ticket_ids=["MESOS-2"]))
    assert checks["retrieval_hit"] is True


def test_invented_citation_is_caught():
    checks = run_checks(_trace(answer=GOOD_ANSWER + " also MESOS-999"))
    assert checks["citations_valid"] is False


def test_abstention_is_correct_only_for_unanswerable_queries():
    abstain = _trace(kind="unanswerable", expected_ticket_ids=[], answer=NO_EVIDENCE_ANSWER)
    assert run_checks(abstain)["abstention_correct"] is True
    assert run_checks(_trace(kind="unanswerable", expected_ticket_ids=[]))["abstention_correct"] is False
    # abstaining on an answerable query is a failure (false abstention)
    assert run_checks(_trace(answer=NO_EVIDENCE_ANSWER))["abstention_correct"] is False


def test_llm_phrased_abstention_is_recognised():
    assert is_abstention("There is not enough supporting evidence to recommend a resolution.")
    assert not is_abstention(GOOD_ANSWER)


def test_irrelevant_evidence_answer_is_recognised_as_abstention():
    # the relevance-gate's decline text, distinct from NO_EVIDENCE_ANSWER -- must still count
    assert is_abstention(IRRELEVANT_EVIDENCE_ANSWER)


def test_upfront_decline_before_a_supporting_evidence_section_is_an_abstention():
    # a genuine LLM-authored decline: states it first, "evidence" section just explains why
    answer = (
        "**Suggested Resolution**\nThere is not enough supporting evidence to recommend a resolution "
        "regarding this specific issue.\n\n**Supporting Evidence**\nThe retrieved evidence does not "
        "address this."
    )
    assert is_abstention(answer)


def test_trailing_hedge_after_a_real_answer_is_not_an_abstention():
    # real, observed failure mode: a full cited answer that ends with a scope-narrowing
    # disclaimer -- this is a hedge, not a decline, and must not be flagged as one
    answer = (
        "**Suggested Resolution:**\n1. Investigate the timing of status updates.\n2. Check for race "
        "conditions similar to MESOS-6026.\n\n**Supporting Evidence:**\n- MESOS-6026 documents this "
        "exact race condition.\n\nThere is not enough supporting evidence to recommend a specific code "
        "change beyond investigating the timing."
    )
    assert not is_abstention(answer)


def test_has_required_sections_correctly_sees_a_trailing_hedge_answer_as_a_real_answer():
    answer = (
        "**Suggested Resolution:**\ndo the thing\n\n**Supporting Evidence:**\ncites MESOS-1\n\n"
        "There is not enough supporting evidence to recommend a specific variant of this fix."
    )
    assert has_required_sections(_trace(answer=answer)) is True


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
