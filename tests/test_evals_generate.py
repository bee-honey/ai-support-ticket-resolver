"""Query generation: seed selection coverage, kind mix, filters, and dataset IO -- LLM mocked."""

from __future__ import annotations

import random
from unittest.mock import MagicMock

from app.models.schemas import RetrievedChunk
from evals.generate_queries import generate_queries, select_seed_chunks
from evals.schemas import load_queries, write_jsonl


def _chunk(ticket: str, component: str, chunk_index: int = 0, text: str = "x" * 300) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata={"ticket_id": ticket, "component": component, "chunk_index": chunk_index, "summary": f"s {ticket}"},
    )


def _corpus() -> list[RetrievedChunk]:
    chunks = [_chunk(f"MESOS-{i}", "docker") for i in range(10)]
    chunks += [_chunk(f"MESOS-{100 + i}", "agent;webui") for i in range(3)]
    chunks += [_chunk("MESOS-200", "storage")]
    chunks += [_chunk("MESOS-1", "docker", chunk_index=1)]            # not a first chunk
    chunks += [_chunk("MESOS-300", "storage", text="too short")]       # too little text
    return chunks


def test_seed_selection_round_robins_components_and_skips_unusable_chunks():
    picked = select_seed_chunks(_corpus(), 3, random.Random(0))
    assert {c.metadata["component"].split(";")[0] for c in picked} == {"docker", "agent", "storage"}
    assert all(c.metadata["chunk_index"] == 0 and len(c.text) >= 200 for c in picked)
    assert "MESOS-300" not in {c.metadata["ticket_id"] for c in picked}


def test_seed_selection_never_duplicates_and_caps_at_available():
    picked = select_seed_chunks(_corpus(), 100, random.Random(0))
    ids = [c.metadata["ticket_id"] for c in picked]
    assert len(ids) == len(set(ids)) == 14


def test_generate_queries_mix_ids_and_filters():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='  "Executors keep dying after restart"  ')
    queries = generate_queries(_corpus(), llm, 10, unanswerable_fraction=0.2, filtered_fraction=0.2, seed=1)

    kinds = [q.kind for q in queries]
    assert (kinds.count("unanswerable"), kinds.count("filtered"), kinds.count("answerable")) == (2, 2, 6)
    assert [q.id for q in queries] == [f"q{i:03d}" for i in range(1, 11)]
    assert all(q.question == "Executors keep dying after restart" for q in queries)  # quotes/space stripped
    assert llm.invoke.call_count == 10  # one call per query

    for q in queries:
        if q.kind == "unanswerable":
            assert q.expected_ticket_id is None and q.filters is None
        else:
            assert q.expected_ticket_id.startswith("MESOS-")
        if q.kind == "filtered":
            assert q.filters and set(q.filters) == {"component"}


def test_generation_is_reproducible_for_a_seed():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="q")
    a = generate_queries(_corpus(), llm, 8, seed=7)
    b = generate_queries(_corpus(), llm, 8, seed=7)
    assert [(q.id, q.kind, q.expected_ticket_id) for q in a] == [(q.id, q.kind, q.expected_ticket_id) for q in b]


def test_dataset_jsonl_roundtrip(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="question text")
    queries = generate_queries(_corpus(), llm, 5, seed=1)
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, queries)
    assert load_queries(path) == queries
