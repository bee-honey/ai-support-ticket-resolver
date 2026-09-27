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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.models.schemas import RAGResult, RAGSource, RetrievedChunk
from app.rag.guardrails import redact_pii
from app.rag.prompts import (
    PROMPT_INJECTION_GATE_PROMPT,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_GATE_PROMPT,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)
from app.retrieval.retriever import Retriever

# How many candidates _retrieve() asks Chroma for internally, before truncating the
# merged, re-sorted result down to the caller's actual requested top_k -- see the
# comment in _retrieve() for why this exists (Chroma's HNSW index measurably misses
# genuine top matches at small n_results values on this corpus; asking for more costs
# single-digit milliseconds regardless).
RETRIEVAL_OVERFETCH_MULTIPLIER = 6
RETRIEVAL_OVERFETCH_MIN = 30

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

# Deliberately vague about WHY -- confirming exactly what tripped the guardrail is free
# information for someone probing it. Returned before retrieval even runs, so there's
# never anything to cite here.
GUARDRAIL_REFUSAL_ANSWER = (
    "I can't help with that request. I'm scoped to answering support questions grounded "
    "in historical tickets and documentation, and I can't change my instructions or share "
    "internal configuration."
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
        gate_model: str | None = None,
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
        # The 3 gates below are independent of `chat_model` on purpose (`gate_model`,
        # defaulting to `settings.gate_model` -- see its definition in
        # app/config/settings.py): they're cheap classification/short-JSON tasks, not
        # the answer itself, so a user picking a stronger/slower chat_model for better
        # answers shouldn't also tax these 3 calls with that model's latency and cost.
        # `max_tokens` is capped too -- each returns a small, bounded JSON object, so
        # there's no legitimate reason for a response to run long, and a runaway one
        # would otherwise silently add latency (autoregressive decoding is O(output
        # tokens)) for zero benefit.
        gate_model_name = gate_model or settings.gate_model

        # A separate, JSON-mode client for the relevance gate below -- it can't share
        # `self._llm`, since forcing JSON mode on that client would break the real
        # (plain-text) answer generation it's also used for.
        self._relevance_llm = ChatOpenAI(
            model=gate_model_name,
            api_key=api_key or settings.openai_api_key,
            temperature=0,
            max_tokens=100,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        # A separate client (not reused from _relevance_llm) mainly so each has its own
        # mock in tests without one call's mock affecting the other. temperature=0, not
        # the original 0.3 ("a little temperature is fine, rewrites are a search aid,
        # not the answer itself") -- that reasoning undersold the real consequence:
        # verified directly that the non-zero temperature was the direct cause of a
        # real user getting two different answers to the identical question. The
        # rewrite decides what gets retrieved, and retrieval decides whether the
        # system answers or declines -- variance here isn't cosmetic, it propagates
        # all the way to the final user-facing outcome. temperature=0 measurably cuts
        # that variance (verified: 4 of 5 repeated calls identical, vs. every single
        # one differing at 0.3) -- not a perfect guarantee (gpt-4o-mini at
        # temperature=0 still isn't bit-perfect, already documented elsewhere in this
        # project), but a real, clear improvement with no offsetting downside.
        self._rewrite_llm = ChatOpenAI(
            model=gate_model_name,
            api_key=api_key or settings.openai_api_key,
            temperature=0,
            max_tokens=100,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        # Its own client too (not reused from _relevance_llm), same reason as
        # _rewrite_llm above: an independent mock per concern in tests, even though
        # both are JSON-mode classifiers under the hood.
        self._injection_llm = ChatOpenAI(
            model=gate_model_name,
            api_key=api_key or settings.openai_api_key,
            temperature=0,
            max_tokens=100,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _is_prompt_injection(self, question: str) -> bool:
        """Cheap gate that runs BEFORE retrieval: is this question actually trying to
        manipulate the assistant (override instructions, extract the system prompt,
        role-play out of scope), as opposed to just an unusual support question?

        Fails OPEN (returns False) on any error or unparseable response, same
        philosophy as the relevance gate. Worth naming explicitly, since it's a
        real trade-off for a *safety* gate specifically: a higher-stakes production
        deployment (real write access, real user PII at risk) would more likely
        fail CLOSED here and block on an uncertain/erroring check instead. This
        project's actual stakes are low -- read-only retrieval, no write access, no
        real user data -- so consistently failing open (matching every other gate
        in this pipeline) is the more defensible choice for it: a flaky guardrail
        call shouldn't take down a legitimate support question.
        """
        try:
            response = self._injection_llm.invoke(
                [SystemMessage(content=PROMPT_INJECTION_GATE_PROMPT), HumanMessage(content=question)]
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            return bool(json.loads(content).get("injection_attempt", False))
        except Exception:
            return False

    def _generate_search_queries(self, question: str) -> list[str]:
        """The original question plus one rewritten variant (always included first,
        as a guaranteed retrieval -- rewriting can only ADD a retrieval attempt here,
        never replace the original with something worse). Targets a failure mode seen
        in eval runs: a user's wording simply not matching how the ticket itself was
        written (measured: retrieval_hit was weaker for questions phrased like a
        ticket's description/resolution than like its summary).

        Scoped to exactly one rewrite (not several): measured against the real eval
        set, 3 rewrites gave a small net gain on retrieval_hit/context_relevance but
        also ~3x'd retrieval latency and correlated with more invalid citations --
        merging results from more, more heterogeneous queries seemed to make it
        harder for the answering LLM to track which evidence backed which claim.
        Cutting to one is a narrower, cheaper bet on the same fix.

        Fails OPEN (returns just [question]) on any error or unparseable response --
        a flaky rewrite call should degrade to the original Phase 1 behaviour (search
        with the question as-is), not block retrieval entirely.
        """
        try:
            response = self._rewrite_llm.invoke(
                [SystemMessage(content=QUERY_REWRITE_PROMPT), HumanMessage(content=question)]
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            raw_queries = json.loads(content).get("queries", [])
            if not isinstance(raw_queries, list):
                raise ValueError("'queries' was not a list")  # e.g. a bare string -- iterating it would give chars
            rewrites = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
        except Exception:
            rewrites = []
        # dict.fromkeys: de-dupe while preserving order (original first); cap at 2 total
        # (original + 1 rewrite) regardless of what came back.
        return list(dict.fromkeys([question, *rewrites]))[:2]

    def _retrieve(
        self, question: str, k: int | None, filters: dict[str, Any] | None
    ) -> tuple[list[RetrievedChunk], float]:
        """Retrieve for the original question AND one rewritten variant IN ONE BATCHED
        call (`Retriever.retrieve_many` -- one embedding request, one Chroma query for
        both variants together, not a round trip per variant), merging results by
        (ticket_id, chunk_index) and keeping each chunk's best score across whichever
        query(s) found it -- so a chunk both queries agree on ranks above one only a
        single query happened to surface. Only the top `top_k` of the merged set is
        returned to the caller -- see the overfetch note below for why more than that
        is asked of Chroma internally.
        """
        started = time.perf_counter()
        top_k = k or get_settings().default_top_k
        queries = self._generate_search_queries(question)

        # Ask Chroma for more candidates than top_k, then truncate to top_k below --
        # verified directly (bypassing this code entirely, straight against the Chroma
        # API) that its HNSW index is NOT reliably accurate at small n_results values on
        # this corpus: a query asked for n_results=5 missed a chunk entirely that an
        # n_results=50 call for the SAME query found at rank 1 with a strong score.
        # Chroma's HNSW search quality scales with how many candidates it's asked to
        # consider, not just with how many you actually want back. The Chroma query
        # itself is single-digit milliseconds regardless of n_results (measured), so
        # this costs nothing in latency -- it only fixes a real recall gap.
        retrieve_k = max(top_k * RETRIEVAL_OVERFETCH_MULTIPLIER, RETRIEVAL_OVERFETCH_MIN)

        best: dict[tuple[Any, Any, str], RetrievedChunk] = {}
        for chunks in self.retriever.retrieve_many(queries, k=retrieve_k, filters=filters):
            for chunk in chunks:
                key = (chunk.metadata.get("ticket_id"), chunk.metadata.get("chunk_index"), chunk.text[:50])
                existing = best.get(key)
                if existing is None or (chunk.score or 0) > (existing.score or 0):
                    best[key] = chunk

        merged = sorted(best.values(), key=lambda c: c.score or 0, reverse=True)[:top_k]
        return merged, time.perf_counter() - started

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
        started = time.perf_counter()
        question = redact_pii(question)

        # The injection check depends on nothing but the (redacted) question, and
        # nothing downstream needs its verdict until right before generation -- so it
        # runs on a background thread for the whole time retrieval + the relevance gate
        # (below) are running, instead of as a 4th sequential round trip in front of
        # them. Measured: the rewrite->retrieve->gate chain alone (~2s) is longer than
        # this check alone (~0.9s), so it's effectively free rather than adding latency.
        # Trade-off: retrieval (and the gate, if evidence came back) now always runs
        # even on an actual injection attempt, since the verdict isn't known until
        # `injection_future.result()` below -- a small wasted cost on the rare case,
        # traded for hiding this check's latency on every legitimate question instead.
        with ThreadPoolExecutor(max_workers=1) as executor:
            injection_future = executor.submit(self._is_prompt_injection, question)

            chunks, _ = self._retrieve(question, k, filters)
            relevant = self._evidence_is_relevant(question, chunks) if chunks else True

            is_injection = injection_future.result()

        retrieval_seconds = time.perf_counter() - started  # the real wall-clock elapsed for the concurrent phase

        if is_injection:
            return RAGResult(answer=GUARDRAIL_REFUSAL_ANSWER, sources=[], retrieval_seconds=retrieval_seconds)

        if not chunks:
            return RAGResult(answer=NO_EVIDENCE_ANSWER, sources=[], retrieval_seconds=retrieval_seconds)

        if not relevant:
            # `chunks` (not just `sources`) stays populated even though we're declining --
            # evals can then tell apart "retrieval missed it" from "retrieval found it but
            # the gate rejected it" instead of both looking like a retrieval failure.
            return RAGResult(
                answer=IRRELEVANT_EVIDENCE_ANSWER, sources=[], chunks=chunks, retrieval_seconds=retrieval_seconds
            )

        gen_started = time.perf_counter()
        response = self._llm.invoke(self._messages(question, chunks))
        generation_seconds = time.perf_counter() - gen_started
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
        started = time.perf_counter()
        question = redact_pii(question)

        # Same concurrent-injection-check design as `answer()` -- see the comment there.
        with ThreadPoolExecutor(max_workers=1) as executor:
            injection_future = executor.submit(self._is_prompt_injection, question)

            chunks, retrieved_seconds = self._retrieve(question, k, filters)
            yield StreamEvent("retrieved", chunks=len(chunks), seconds=retrieved_seconds)

            relevant = self._evidence_is_relevant(question, chunks) if chunks else True

            is_injection = injection_future.result()

        retrieval_seconds = time.perf_counter() - started  # the real wall-clock elapsed for the concurrent phase

        if is_injection:
            yield StreamEvent(
                "done",
                result=RAGResult(answer=GUARDRAIL_REFUSAL_ANSWER, sources=[], retrieval_seconds=retrieval_seconds),
            )
            return

        if not chunks:
            yield StreamEvent(
                "done",
                result=RAGResult(answer=NO_EVIDENCE_ANSWER, sources=[], retrieval_seconds=retrieval_seconds),
            )
            return

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
        gen_started = time.perf_counter()
        for piece in self._llm.stream(self._messages(question, chunks)):
            text = piece.content if isinstance(piece.content, str) else ""
            if text:
                parts.append(text)
                yield StreamEvent("token", text=text)
            if isinstance(getattr(piece, "usage_metadata", None), dict):
                usage = piece  # only the final chunk carries usage
        generation_seconds = time.perf_counter() - gen_started
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
