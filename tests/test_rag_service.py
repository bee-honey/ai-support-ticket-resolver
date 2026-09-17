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
