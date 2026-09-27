"""Log every live chat answer as a `Trace` -- the exact shape eval runs already
use -- so a thumbs-down given in the chat UI (see `ui/views/chat.py`) has real
question/answer/evidence to point at later, instead of being a bare,
uninspectable pass/fail sitting in a labels file on its own.

Not an eval run: there's no `expected_ticket_ids`/`kind` ground truth for a
live question someone actually typed, so deterministic checks and judges
don't apply here and `checks`/`judgments` stay empty. This is a production
trace (real usage), not a test-set trace (known-correct answers) -- kept in
its own file (`evals/chat_logs/log.jsonl`, not `evals/results/`) so it's never
mistaken for a labeled run in the Evals page's run picker, and so
`evals/results/*.jsonl` stays exactly what `--no-judge`/CI-style tooling
already assumes it is: eval runs only.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.models.schemas import RAGResult
from app.rag.guardrails import redact_pii
from app.rag.prompts import build_context
from app.rag.service import is_abstention
from evals.schemas import Trace, append_jsonl, load_labels, load_traces

CHAT_RUN_ID = "chat"
DEFAULT_CHAT_LOG = Path("evals/chat_logs/log.jsonl")


def log_chat_turn(
    question: str,
    result: RAGResult,
    *,
    chat_model: str,
    top_k: int | None,
    log_path: Path | str = DEFAULT_CHAT_LOG,
) -> Trace:
    """Persist one live chat answer as a `Trace` and return it -- the caller
    (the chat UI) keeps `trace.trace_id` to attach feedback to via `HumanLabel`.

    Redacts `question` again here rather than trusting the caller to pass an
    already-redacted string: `RAGService.answer`/`stream_answer` redact their
    OWN local copy internally (see app/rag/service.py) and that never
    propagates back to the caller's variable, so the chat UI's own `question`
    is still the raw, unredacted text. `redact_pii` is cheap and idempotent,
    so enforcing it again at this specific boundary -- the one place PII
    actually gets written to disk -- costs nothing and closes that gap for
    good, regardless of what any future caller passes in.
    """
    trace = Trace(
        run_id=CHAT_RUN_ID,
        query_id=uuid.uuid4().hex[:12],
        question=redact_pii(question),
        kind="live",  # not a QUERY_KINDS value on purpose -- this isn't from a labeled test set
        chat_model=chat_model,
        top_k=top_k,
        answer=result.answer,
        context=build_context(result.chunks) if result.chunks else "",
        retrieved_ticket_ids=[c.ticket_id for c in result.chunks if c.ticket_id],
        retrieval_seconds=result.retrieval_seconds,
        generation_seconds=result.generation_seconds,
        total_seconds=result.total_seconds,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    append_jsonl(log_path, trace)
    return trace


def hot_issues(
    log_path: Path | str = DEFAULT_CHAT_LOG,
    feedback_path: Path | str | None = None,
    limit: int = 5,
) -> list[str]:
    """The most-asked distinct questions in the live chat log, most-frequent first
    (ties broken by recency) -- feeds the chat page's "Recent Hot Issues" panel.

    A "hot issue" suggestion is never one the system is already known to handle
    badly: a question is excluded if its most recent logged answer declined
    (`is_abstention`) or if its feedback verdict was "fail" (see `feedback_path`,
    the chat feedback labels file) -- there's no point surfacing a question as a
    shortcut into an answer that's already known to be wrong.

    Returns [] if the log doesn't exist yet, is empty, or every question was
    excluded -- the caller (`ui/views/chat.py`) decides what to show instead
    until there's enough real traffic for this to be interesting.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    traces = load_traces(path)
    if not traces:
        return []
    labels = load_labels(feedback_path) if feedback_path else {}

    # normalized question text -> running stats, so "Docker fails" and "docker
    # fails" (different casing/whitespace, same underlying question) count as one
    groups: dict[str, dict] = {}
    for index, trace in enumerate(traces):
        key = trace.question.strip().lower()
        if not key:
            continue
        label = labels.get(trace.trace_id)
        is_bad_answer = is_abstention(trace.answer) or (label is not None and label.verdict == "fail")
        group = groups.setdefault(key, {"display": trace.question, "count": 0, "order": index, "bad": False})
        group["count"] += 1
        group["order"] = index  # keep the most recent occurrence's position, for the recency tiebreak
        group["display"] = trace.question  # keep the most recent phrasing/casing shown to the user
        group["bad"] = group["bad"] or is_bad_answer

    candidates = [g for g in groups.values() if not g["bad"]]
    candidates.sort(key=lambda g: (g["count"], g["order"]), reverse=True)
    return [g["display"] for g in candidates[:limit]]
