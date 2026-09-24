"""Deterministic (no-LLM) checks over a trace.

These are the cheapest layer of the eval stack, so they run first. Each check
returns True (pass), False (fail) or None (not applicable to this trace).
"""

from __future__ import annotations

import re

from app.rag.service import NO_EVIDENCE_ANSWER
from evals.schemas import Trace

DEFAULT_LATENCY_CAP_SECONDS = 10.0

# Ticket IDs look like MESOS-1234 / SPARK-99 (PROJECT-number).
TICKET_ID_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")

# The system prompt's preferred refusal wording; NO_EVIDENCE_ANSWER contains it too.
_ABSTAIN_MARKER = "not enough supporting evidence"
REQUIRED_SECTIONS = ("Suggested Resolution", "Supporting Evidence")


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return _ABSTAIN_MARKER in lowered or answer.strip() == NO_EVIDENCE_ANSWER.strip()


def cited_ticket_ids(answer: str) -> set[str]:
    return set(TICKET_ID_PATTERN.findall(answer))


def retrieval_hit(trace: Trace) -> bool | None:
    """Was any one of this query's expected tickets among the retrieved ones?

    Some queries have more than one equally valid source ticket (e.g. a
    duplicate-triage case that could correctly cite either the original ticket
    or a later duplicate of it) -- any single hit counts.
    """
    if not trace.expected_ticket_ids:
        return None
    return any(ticket_id in trace.retrieved_ticket_ids for ticket_id in trace.expected_ticket_ids)


def citations_valid(trace: Trace) -> bool | None:
    """Every ticket ID cited in the answer must have actually been retrieved."""
    if trace.error:
        return None
    return cited_ticket_ids(trace.answer) <= set(trace.retrieved_ticket_ids)


def abstention_correct(trace: Trace) -> bool | None:
    """Unanswerable queries must be declined; answerable ones must not be."""
    if trace.error:
        return None
    abstained = is_abstention(trace.answer)
    return abstained if trace.kind == "unanswerable" else not abstained


def has_required_sections(trace: Trace) -> bool | None:
    if trace.error or is_abstention(trace.answer):
        return None
    return all(section.lower() in trace.answer.lower() for section in REQUIRED_SECTIONS)


def within_latency(trace: Trace, cap_seconds: float = DEFAULT_LATENCY_CAP_SECONDS) -> bool | None:
    if trace.error:
        return None
    return trace.total_seconds <= cap_seconds


def run_checks(trace: Trace, latency_cap_seconds: float = DEFAULT_LATENCY_CAP_SECONDS) -> dict[str, bool | None]:
    return {
        "retrieval_hit": retrieval_hit(trace),
        "citations_valid": citations_valid(trace),
        "abstention_correct": abstention_correct(trace),
        "has_required_sections": has_required_sections(trace),
        "within_latency": within_latency(trace, latency_cap_seconds),
    }
