"""RAG service: the single entry point for "answer this support problem".

Deliberately UI-agnostic. Today it's called from Streamlit
(`ui/streamlit_app.py`); in Phase 2 the same `RAGService.answer(...)` call
can sit behind a FastAPI endpoint or be wrapped as a LangGraph tool/node
without changing retrieval or prompting logic.
"""

from __future__ import annotations

import time
from typing import Any

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
        )

    def answer(
        self,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResult:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(question, k=k, filters=filters)
        retrieval_seconds = time.perf_counter() - started

        if not chunks:
            return RAGResult(answer=NO_EVIDENCE_ANSWER, sources=[], retrieval_seconds=retrieval_seconds)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(question, chunks)),
        ]
        started = time.perf_counter()
        response = self._llm.invoke(messages)
        generation_seconds = time.perf_counter() - started
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        input_tokens, output_tokens = _token_counts(response)

        return RAGResult(
            answer=answer_text,
            sources=_dedupe_sources(chunks),
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
