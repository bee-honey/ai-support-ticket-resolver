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

import time
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from app.models.schemas import RAGResult, RAGSource, RetrievedChunk
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import Retriever

NO_EVIDENCE_ANSWER = (
    "There is not enough supporting evidence to recommend a resolution. "
    "No historical tickets or documentation relevant to this problem were found."
)


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

    def _retrieve(
        self, question: str, k: int | None, filters: dict[str, Any] | None
    ) -> tuple[list[RetrievedChunk], float]:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(question, k=k, filters=filters)
        return chunks, time.perf_counter() - started

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

        started = time.perf_counter()
        response = self._llm.invoke(self._messages(question, chunks))
        generation_seconds = time.perf_counter() - started
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        input_tokens, output_tokens = _token_counts(response)

        return RAGResult(
            answer=answer_text,
            sources=_dedupe_sources(chunks),
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

        yield StreamEvent(
            "done",
            result=RAGResult(
                answer="".join(parts),
                sources=_dedupe_sources(chunks),
                chunks=chunks,
                retrieval_seconds=retrieval_seconds,
                generation_seconds=generation_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )
