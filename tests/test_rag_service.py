"""RAGService: no-evidence fallback, source dedupe, prompt grounding -- LLM is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.models.schemas import RetrievedChunk
from app.rag.service import NO_EVIDENCE_ANSWER, RAGService


def _make_service(chunks, llm_content: str = "answer text"):
    service = RAGService.__new__(RAGService)  # skip __init__: no real ChatOpenAI client
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    service.retriever = retriever
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=llm_content)
    service._llm = llm
    return service, llm


def test_no_evidence_short_circuits_without_calling_llm():
    service, llm = _make_service([])
    result = service.answer("some question")
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.sources == []
    llm.invoke.assert_not_called()


def test_sources_are_deduplicated_by_ticket_id_in_retrieval_order():
    chunks = [
        RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "component": "docker", "chunk_index": 0}),
        RetrievedChunk(text="b", metadata={"ticket_id": "T-1", "component": "docker", "chunk_index": 1}),
        RetrievedChunk(text="c", metadata={"ticket_id": "T-2", "component": "networking", "chunk_index": 0}),
    ]
    service, _ = _make_service(chunks)
    result = service.answer("q")
    assert [s.ticket_id for s in result.sources] == ["T-1", "T-2"]


def test_prompt_sent_to_llm_includes_all_retrieved_ticket_ids():
    chunks = [
        RetrievedChunk(text="Docker daemon fix", metadata={"ticket_id": "T-1", "component": "docker", "chunk_index": 0}),
        RetrievedChunk(text="Network fix", metadata={"ticket_id": "T-2", "component": "networking", "chunk_index": 0}),
    ]
    service, llm = _make_service(chunks)
    service.answer("q")
    messages = llm.invoke.call_args[0][0]
    user_prompt = messages[1].content
    assert "T-1" in user_prompt
    assert "T-2" in user_prompt


def test_result_reports_token_usage_and_timing_from_llm_metadata():
    chunks = [RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    service, llm = _make_service(chunks)
    llm.invoke.return_value = MagicMock(
        content="answer", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    )
    result = service.answer("q")
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (120, 30, 150)
    assert result.retrieval_seconds >= 0
    assert result.generation_seconds >= 0
    assert result.total_seconds == result.retrieval_seconds + result.generation_seconds


def test_token_usage_is_none_when_provider_does_not_report_it():
    chunks = [RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    service, _ = _make_service(chunks)  # MagicMock response: usage_metadata is not a dict
    result = service.answer("q")
    assert result.input_tokens is None
    assert result.total_tokens is None


def test_no_evidence_result_has_retrieval_time_but_no_tokens():
    service, _ = _make_service([])
    result = service.answer("q")
    assert result.generation_seconds == 0.0
    assert result.total_tokens is None


def test_result_exposes_the_raw_retrieved_chunks_for_evals():
    chunks = [
        RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0}),
        RetrievedChunk(text="b", metadata={"ticket_id": "T-1", "chunk_index": 1}),
    ]
    service, _ = _make_service(chunks)
    result = service.answer("q")
    assert result.chunks == chunks  # not deduped, unlike result.sources


# ---- streaming ---------------------------------------------------------------------------------


def _stream_service(chunks, pieces):
    """Service whose LLM `.stream()` yields `pieces` (objects with .content / .usage_metadata)."""
    service, llm = _make_service(chunks)
    llm.stream.return_value = iter(pieces)
    return service, llm


def _piece(text="", usage=None):
    return MagicMock(content=text, usage_metadata=usage)


def test_stream_answer_yields_retrieved_then_tokens_then_done_with_exact_usage():
    chunks = [RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    pieces = [_piece("Hel"), _piece("lo"), _piece("", usage={"input_tokens": 90, "output_tokens": 2, "total_tokens": 92})]
    service, _ = _stream_service(chunks, pieces)

    events = list(service.stream_answer("q"))

    assert [e.kind for e in events] == ["retrieved", "token", "token", "done"]
    assert events[0].chunks == 1 and events[0].seconds >= 0
    assert "".join(e.text for e in events if e.kind == "token") == "Hello"
    result = events[-1].result
    assert result.answer == "Hello"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (90, 2, 92)
    assert [s.ticket_id for s in result.sources] == ["T-1"] and result.chunks == chunks
    assert result.generation_seconds >= 0 and result.retrieval_seconds >= 0


def test_stream_answer_without_evidence_skips_the_llm():
    service, llm = _stream_service([], [])
    events = list(service.stream_answer("q"))
    assert [e.kind for e in events] == ["retrieved", "done"]
    assert events[0].chunks == 0 and events[-1].result.answer == NO_EVIDENCE_ANSWER
    llm.stream.assert_not_called()


def test_stream_answer_sends_the_same_prompt_as_answer():
    chunks = [RetrievedChunk(text="Docker fix", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    service, llm = _stream_service(chunks, [_piece("x")])
    list(service.stream_answer("what now"))
    streamed = llm.stream.call_args[0][0]
    service.answer("what now")
    invoked = llm.invoke.call_args[0][0]
    assert [m.content for m in streamed] == [m.content for m in invoked]


def test_stream_answer_leaves_usage_none_when_the_provider_omits_it():
    chunks = [RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    service, _ = _stream_service(chunks, [_piece("hi")])
    result = list(service.stream_answer("q"))[-1].result
    assert result.total_tokens is None


def test_stream_answer_propagates_llm_errors():
    import pytest

    chunks = [RetrievedChunk(text="a", metadata={"ticket_id": "T-1", "chunk_index": 0})]
    service, llm = _make_service(chunks)
    llm.stream.side_effect = RuntimeError("rate limited")
    with pytest.raises(RuntimeError, match="rate limited"):
        list(service.stream_answer("q"))
