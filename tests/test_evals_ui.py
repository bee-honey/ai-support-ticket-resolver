"""Evals Streamlit page, driven headlessly with AppTest against a temp evals dir (no API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from app.config.settings import model_choices
from app.models.schemas import RAGResult, RetrievedChunk
from evals.schemas import (
    EvalQuery,
    JudgeVerdict,
    Trace,
    load_labels,
    queries_to_rows,
    read_jsonl,
    rows_to_queries,
    write_jsonl,
)

PAGE = str(Path(__file__).resolve().parent.parent / "ui" / "pages" / "1_Evals.py")
GOOD = "Suggested Resolution\n- fix\n\nSupporting Evidence\n- MESOS-1"


def _trace(run_id: str, query_id: str, faithful: int, seconds: float = 2.0, kind: str = "answerable") -> Trace:
    trace = Trace(
        run_id=run_id, query_id=query_id, question=f"question {query_id}", kind=kind, chat_model="gpt-4o-mini",
        top_k=5, expected_ticket_id="MESOS-1", answer=GOOD, context="[Evidence 1] ticket_id=MESOS-1\nfix it",
        retrieved_ticket_ids=["MESOS-1"], retrieval_seconds=0.2, generation_seconds=seconds - 0.2,
        total_seconds=seconds, input_tokens=100, output_tokens=20,
    )
    trace.checks = {"retrieval_hit": True, "citations_valid": True, "within_latency": True}
    trace.judgments = {
        "faithfulness": JudgeVerdict(faithful, "supported" if faithful else "invented a flag"),
        "answer_relevancy": JudgeVerdict(1, "on topic"),
    }
    return trace


@pytest.fixture
def evals_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("EVALS_DIR", str(tmp_path))
    st_cache = __import__("streamlit").cache_resource
    st_cache.clear()
    write_jsonl(tmp_path / "results/20260101-000001-gpt-4o-mini.jsonl",
                [_trace("20260101-000001-gpt-4o-mini", "q1", 1), _trace("20260101-000001-gpt-4o-mini", "q2", 0)])
    write_jsonl(tmp_path / "results/20260102-000001-gpt-4o-mini.jsonl",
                [_trace("20260102-000001-gpt-4o-mini", "q1", 1, seconds=4.0), _trace("20260102-000001-gpt-4o-mini", "q2", 1, seconds=4.0)])
    return tmp_path


def _page() -> AppTest:
    return AppTest.from_file(PAGE, default_timeout=30).run()


def _button(at: AppTest, prefix: str):
    return next(b for b in at.button if b.label.startswith(prefix))


def _select(at: AppTest, label: str):
    return next(s for s in at.selectbox if s.label == label)


def test_page_renders_metric_tiles_for_the_newest_run(evals_dir):
    at = _page()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["▶ Run", "📈 Metrics", "🏷 Label", "🎯 Align"]
    assert _select(at, "Selected run").value == "20260102-000001-gpt-4o-mini.jsonl"  # newest first
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Queries"] == "2" and tiles["faithfulness"] == "100%" and tiles["Latency p50"] == "4.0s"
    assert tiles["Tool calls / query"] == "n/a"


def test_selecting_the_older_run_shows_its_own_numbers(evals_dir):
    at = _page()
    _select(at, "Selected run").select("20260101-000001-gpt-4o-mini.jsonl").run()
    assert {m.label: m.value for m in at.metric}["faithfulness"] == "50%"


def test_compare_shows_deltas_with_latency_inverted(evals_dir):
    at = _page()
    _select(at, "Compare with").select("20260101-000001-gpt-4o-mini.jsonl").run()
    tiles = {m.label: m for m in at.metric}
    assert tiles["faithfulness"].delta == "+50 pts"   # 100% vs 50%
    assert tiles["Latency p50"].delta == "+2.0s"       # slower than before


def test_empty_state_points_to_the_run_tab(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALS_DIR", str(tmp_path))
    at = _page()
    assert not at.exception
    assert any("No runs yet" in i.value for i in at.info)


def test_labeling_saves_a_label_and_moves_to_the_next_trace(evals_dir):
    at = _page()
    assert _select(at, "Trace").value.endswith(":q1")
    at.radio[0].set_value("fail")
    next(t for t in at.text_input if t.label == "Reason (short)").set_value("steps not in evidence")
    _button(at, "Save & next").click().run()

    labels = load_labels(evals_dir / "labels/human_labels.jsonl")
    label = labels["20260102-000001-gpt-4o-mini:q1"]
    assert (label.verdict, label.reason) == ("fail", "steps not in evidence")
    assert _select(at, "Trace").value.endswith(":q2")  # advanced to the next unlabeled trace


def test_align_tab_compares_judges_with_labels_and_lists_disagreements(evals_dir):
    from evals.schemas import HumanLabel, append_jsonl

    run = "20260102-000001-gpt-4o-mini"
    append_jsonl(evals_dir / "labels/human_labels.jsonl", HumanLabel(f"{run}:q1", "pass", "fine"))
    append_jsonl(evals_dir / "labels/human_labels.jsonl", HumanLabel(f"{run}:q2", "fail", "wrong step"))
    at = _page()
    assert not at.exception
    overall = next(df.value for df in at.dataframe if "agreement" in df.value.columns)
    row = overall[overall["metric"] == "overall"].iloc[0]
    assert row["labeled"] == 2 and row["FP"] == 1      # judge passed the answer the human failed
    assert any("disagreement" in e.label for e in at.expander)


def test_saving_a_judge_prompt_writes_the_markdown_file(evals_dir, monkeypatch, tmp_path):
    import evals.judges.base as judges_base

    fake_dir = tmp_path / "prompts"
    fake_dir.mkdir()
    (fake_dir / "context_relevance.md").write_text("old prompt")
    monkeypatch.setattr(judges_base, "PROMPTS_DIR", fake_dir)
    at = _page()
    next(t for t in at.text_area if t.label == "System prompt").set_value("new prompt text")
    _button(at, "Save prompt").click().run()
    assert (fake_dir / "context_relevance.md").read_text().strip() == "new prompt text"


def test_run_flow_creates_a_run_file_with_traces_checks_and_judgments(evals_dir, monkeypatch):
    import app.rag.service as rag_service
    import app.retrieval.retriever as retriever
    import evals.judges.base as judges_base

    write_jsonl(evals_dir / "datasets/queries.jsonl", [EvalQuery(id="q1", question="docker fails"),
                                                       EvalQuery(id="q2", question="terraform", kind="unanswerable")])

    class FakeService:
        def __init__(self, retriever=None, chat_model=None, **_):
            self.chat_model = chat_model

        def answer(self, question, k=None, filters=None):
            chunk = RetrievedChunk(text="evidence", metadata={"ticket_id": "MESOS-1", "chunk_index": 0})
            return RAGResult(answer=GOOD, chunks=[chunk], retrieval_seconds=0.1, generation_seconds=0.5,
                             input_tokens=50, output_tokens=10)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content='{"score": 1, "reason": "ok"}')
    monkeypatch.setattr(rag_service, "RAGService", FakeService)
    monkeypatch.setattr(retriever, "Retriever", lambda: MagicMock())
    monkeypatch.setattr(judges_base, "ChatOpenAI", lambda **_: fake_llm)

    at = _page()
    _select(at, "Chat model (answers the queries)").select("gpt-4o")
    _button(at, "▶ Run evals").click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("finished (2 queries)" in s.value for s in at.success)
    new_runs = [p for p in (evals_dir / "results").glob("*.jsonl") if p.name.endswith("-gpt-4o.jsonl")]
    assert len(new_runs) == 1
    rows = read_jsonl(new_runs[0])
    assert [r["chat_model"] for r in rows] == ["gpt-4o", "gpt-4o"]
    assert rows[0]["checks"]["citations_valid"] is True
    assert rows[0]["judgments"]["faithfulness"]["score"] == 1
    assert _select(at, "Selected run").value == new_runs[0].name  # the new run is auto-selected


def test_rows_roundtrip_assigns_missing_ids_and_drops_blank_rows():
    rows = queries_to_rows([EvalQuery(id="q001", question="a", filters={"component": "docker"}, kind="filtered")])
    assert rows[0]["filters"] == '{"component": "docker"}'
    rows += [{"id": "", "question": "new one", "kind": "answerable"}, {"id": "", "question": "  "}]
    queries = rows_to_queries(rows)
    assert [q.id for q in queries] == ["q001", "q002"]
    assert queries[0].filters == {"component": "docker"} and queries[1].filters is None


@pytest.mark.parametrize(
    "row, message",
    [
        ({"id": "a", "question": "q", "filters": "{bad"}, "not valid JSON"),
        ({"id": "a", "question": "q", "filters": "[1]"}, "JSON object"),
        ({"id": "a", "question": "q", "kind": "weird"}, "kind must be"),
    ],
)
def test_rows_to_queries_rejects_bad_input(row, message):
    with pytest.raises(ValueError, match=message):
        rows_to_queries([row])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        rows_to_queries([{"id": "a", "question": "x"}, {"id": "a", "question": "y"}])


def test_model_choices_always_includes_the_current_model():
    assert "my-custom-model" in model_choices("my-custom-model")
    assert model_choices("gpt-4o-mini").count("gpt-4o-mini") == 1
