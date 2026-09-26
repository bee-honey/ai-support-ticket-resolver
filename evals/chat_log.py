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
from evals.schemas import Trace, append_jsonl

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
