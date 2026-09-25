"""RAG service: the single entry point for "answer this support problem".

Deliberately UI-agnostic. Today it's called from Streamlit
(`ui/views/chat.py`); in Phase 2 the same `RAGService.answer(...)` call
can sit behind a FastAPI endpoint or be wrapped as a LangGraph tool/node
without changing retrieval or prompting logic.

Two entry points share the same retrieval and prompting:
  - `answer()`         one blocking call -> `RAGResult` (evals, scripts, tests)
  - `stream_answer()`  yields events as work progresses so a UI can show live
                       progress; the last event carries the same `RAGResult`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.models.schemas import RAGResult, RAGSource, RetrievedChunk
from app.rag.prompts import RELEVANCE_GATE_PROMPT, SYSTEM_PROMPT, build_context, build_user_prompt
from app.retrieval.retriever import Retriever

NO_EVIDENCE_ANSWER = (
    "There is not enough supporting evidence to recommend a resolution. "
    "No historical tickets or documentation relevant to this problem were found."
)

# Distinct from NO_EVIDENCE_ANSWER, which claims nothing was found at all -- that's
# false when the relevance gate declines: retrieval DID return chunks, they were just
# judged not relevant. Reusing NO_EVIDENCE_ANSWER's wording here was a real bug, caught
# by the faithfulness eval judge itself, which correctly flagged "the answer claims no
# tickets were found, but tickets are retrieved" as an unsupported claim.
IRRELEVANT_EVIDENCE_ANSWER = (
    "There is not enough supporting evidence to recommend a resolution. "
    "The retrieved historical tickets and documentation do not appear relevant to this problem."
)

# The system prompt's suggested refusal wording; both answers above contain it too.
_ABSTAIN_MARKER = "not enough supporting evidence"


def is_abstention(answer: str) -> bool:
    """True if `answer` is fundamentally declining, not just hedging at the end.

    Canonical home for this check: it's app-level (deciding whether to show
    sources next to an answer that isn't really one), and `evals/checks.py`
    imports it from here rather than keeping its own copy, so there's exactly
    one definition of "did this answer actually decline".

    A real decline states it up front, or is one of the two canned answers
    above. Observed real failure mode: the model writes a full, cited
    "Suggested Resolution" + "Supporting Evidence" answer, then tacks on a
    trailing disclaimer like "there is not enough supporting evidence to
    recommend a *specific* code change beyond this" -- that's a hedge on an
    otherwise real answer, not a decline. Distinguished by position: the
    marker only counts if it appears before any "Supporting Evidence" section
    (a real decline never gets that far).
    """
    stripped = answer.strip()
    if stripped == NO_EVIDENCE_ANSWER.strip() or stripped == IRRELEVANT_EVIDENCE_ANSWER.strip():
        return True
    lowered = answer.lower()
    marker_index = lowered.find(_ABSTAIN_MARKER)
    if marker_index == -1:
        return False
    evidence_heading_index = lowered.find("supporting evidence")
    return evidence_heading_index == -1 or marker_index <= evidence_heading_index


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[RAGSource]:
    """Collapse chunks into one citation per ticket/document, in retrieval order."""
    seen: set[str] = set()
    sources: list[RAGSource] = []
    for chunk in chunks:
        key = chunk.metadata.get("ticket_id") or chunk.metadata.get("source_file") or chunk.text[:50]
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            RAGSource(
                ticket_id=chunk.metadata.get("ticket_id"),
                component=chunk.metadata.get("component"),
                status=chunk.metadata.get("status"),
                summary=chunk.metadata.get("summary"),
                resolved_date=chunk.metadata.get("resolved_date"),
                source_file=chunk.metadata.get("source_file"),
            )
        )
    return sources


def _token_counts(response: Any) -> tuple[int | None, int | None]:
    """(input, output) token counts from a LangChain message, or None if unreported."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None, None
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    return (
        input_tokens if isinstance(input_tokens, int) else None,
        output_tokens if isinstance(output_tokens, int) else None,
    )


@dataclass
class StreamEvent:
    """One progress event from `RAGService.stream_answer`.

    kind="retrieved": retrieval finished (`chunks` found in `seconds`).
    kind="token":     `text` is the next piece of the answer.
    kind="done":      `result` is the complete `RAGResult` (exact timings and tokens).
    """

    kind: str
    text: str = ""
    chunks: int = 0
    seconds: float = 0.0
    result: RAGResult | None = None


class RAGService:
    def __init__(
        self,
        retriever: Retriever | None = None,
        chat_model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.retriever = retriever or Retriever()
        self._llm = ChatOpenAI(
            model=chat_model or settings.chat_model,
            api_key=api_key or settings.openai_api_key,
            temperature=0,
            stream_usage=True,  # so a streamed answer still reports exact token counts
        )
        # A separate, JSON-mode client for the relevance gate below -- it can't share
        # `self._llm`, since forcing JSON mode on that client would break the real
        # (plain-text) answer generation it's also used for.
        self._relevance_llm = ChatOpenAI(
            model=chat_model or settings.chat_model,
            api_key=api_key or settings.openai_api_key,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _retrieve(
        self, question: str, k: int | None, filters: dict[str, Any] | None
    ) -> tuple[list[RetrievedChunk], float]:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(question, k=k, filters=filters)
        return chunks, time.perf_counter() - started

    def _evidence_is_relevant(self, question: str, chunks: list[RetrievedChunk]) -> bool:
        """Cheap gate before committing to a full generation call: is the retrieved
        evidence actually about this problem, or just superficially similar?

        A raw similarity-score threshold was tried first and measurably failed on
        real data (a nonexistent-ticket-ID query scored higher than 17 of 36 real
        answerable questions, purely from sharing ticket-ID-shaped vocabulary) --
        this needs judgment, not a cutoff.

        Fails OPEN (returns True) on any error or unparseable response, so a flaky
        gate call degrades to the original Phase 1 behavior -- let generation
        decide -- rather than silently refusing to answer a possibly-good question.
        """
        messages = [
            SystemMessage(content=RELEVANCE_GATE_PROMPT),
            HumanMessage(content=f"Problem: {question}\n\nEvidence:\n{build_context(chunks)}"),
        ]
        try:
            response = self._relevance_llm.invoke(messages)
            content = response.content if isinstance(response.content, str) else str(response.content)
            return bool(json.loads(content).get("relevant", True))
        except Exception:
            return True

    @staticmethod
    def _messages(question: str, chunks: list[RetrievedChunk]) -> list[Any]:
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(question, chunks)),
        ]

    def answer(
        self,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResult:
        chunks, retrieval_seconds = self._retrieve(question, k, filters)

        if not chunks:
            return RAGResult(answer=NO_EVIDENCE_ANSWER, sources=[], retrieval_seconds=retrieval_seconds)

        gate_started = time.perf_counter()
        relevant = self._evidence_is_relevant(question, chunks)
        retrieval_seconds += time.perf_counter() - gate_started  # counted as part of the pre-generation phase

        if not relevant:
            # `chunks` (not just `sources`) stays populated even though we're declining --
            # evals can then tell apart "retrieval missed it" from "retrieval found it but
            # the gate rejected it" instead of both looking like a retrieval failure.
            return RAGResult(
                answer=IRRELEVANT_EVIDENCE_ANSWER, sources=[], chunks=chunks, retrieval_seconds=retrieval_seconds
            )

        started = time.perf_counter()
        response = self._llm.invoke(self._messages(question, chunks))
        generation_seconds = time.perf_counter() - started
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        input_tokens, output_tokens = _token_counts(response)

        return RAGResult(
            answer=answer_text,
            # The gate approving evidence doesn't guarantee generation itself won't
            # still decide to decline (its own judgment, per the system prompt) --
            # sources are only meaningful next to an answer that actually used them.
            sources=[] if is_abstention(answer_text) else _dedupe_sources(chunks),
            chunks=chunks,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def stream_answer(
        self,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Same work as `answer()`, but yields progress: retrieved -> token* -> done.

        Exceptions (network, auth, ...) propagate to the caller, as with `answer()`.
        """
        chunks, retrieval_seconds = self._retrieve(question, k, filters)
        yield StreamEvent("retrieved", chunks=len(chunks), seconds=retrieval_seconds)

        if not chunks:
            yield StreamEvent(
                "done",
                result=RAGResult(answer=NO_EVIDENCE_ANSWER, sources=[], retrieval_seconds=retrieval_seconds),
            )
            return

        gate_started = time.perf_counter()
        relevant = self._evidence_is_relevant(question, chunks)
        retrieval_seconds += time.perf_counter() - gate_started

        if not relevant:
            yield StreamEvent(
                "done",
                result=RAGResult(
                    answer=IRRELEVANT_EVIDENCE_ANSWER, sources=[], chunks=chunks, retrieval_seconds=retrieval_seconds
                ),
            )
            return

        parts: list[str] = []
        usage: Any = None
        started = time.perf_counter()
        for piece in self._llm.stream(self._messages(question, chunks)):
            text = piece.content if isinstance(piece.content, str) else ""
            if text:
                parts.append(text)
                yield StreamEvent("token", text=text)
            if isinstance(getattr(piece, "usage_metadata", None), dict):
                usage = piece  # only the final chunk carries usage
        generation_seconds = time.perf_counter() - started
        input_tokens, output_tokens = _token_counts(usage)

        final_answer = "".join(parts)
        yield StreamEvent(
            "done",
            result=RAGResult(
                answer=final_answer,
                sources=[] if is_abstention(final_answer) else _dedupe_sources(chunks),
                chunks=chunks,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=generation_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
