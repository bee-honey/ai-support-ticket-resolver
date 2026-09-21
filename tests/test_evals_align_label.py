"""Judge/human alignment maths and the terminal labeling flow."""

from __future__ import annotations

from evals.align import align, render_alignment
from evals.label import label_traces
from evals.schemas import HumanLabel, JudgeVerdict, Trace, load_labels


def _trace(query_id: str, faith: int | None, rel: int | None = 1, **kw) -> Trace:
    trace = Trace(run_id="r", query_id=query_id, question="q", kind="answerable", chat_model="m", top_k=5,
                  answer="ans", context="[Evidence 1]\nctx", **kw)
    if faith is not None:
        trace.judgments["faithfulness"] = JudgeVerdict(faith, f"faith {faith}")
    if rel is not None:
        trace.judgments["answer_relevancy"] = JudgeVerdict(rel, f"rel {rel}")
    return trace


def _label(query_id: str, verdict: str, **metrics: int) -> HumanLabel:
    return HumanLabel(trace_id=f"r:{query_id}", verdict=verdict, reason=f"{verdict} because", metrics=metrics)


def test_confusion_matrix_positive_is_pass():
    traces = [_trace("a", 1), _trace("b", 0), _trace("c", 1), _trace("d", 0)]
    labels = {l.trace_id: l for l in [
        _label("a", "pass", faithfulness=1),   # TP
        _label("b", "fail", faithfulness=0),   # TN
        _label("c", "fail", faithfulness=0),   # FP: judge lenient
        _label("d", "pass", faithfulness=1),   # FN: judge strict
    ]}
    faith = next(a for a in align(traces, labels) if a.metric == "faithfulness")
    assert (faith.tp, faith.tn, faith.fp, faith.fn) == (1, 1, 1, 1)
    assert faith.agreement == 0.5 and faith.precision == 0.5 and faith.recall == 0.5
    assert {d.trace.query_id for d in faith.disagreements} == {"c", "d"}


def test_overall_is_and_of_answer_judges_vs_human_verdict():
    traces = [_trace("a", 1, rel=1), _trace("b", 1, rel=0), _trace("c", None, rel=1)]
    labels = {l.trace_id: l for l in [_label("a", "pass"), _label("b", "fail"), _label("c", "pass")]}
    overall = next(a for a in align(traces, labels) if a.metric == "overall")
    assert (overall.tp, overall.tn) == (1, 1) and overall.n == 2  # "c" has an unscored judge -> skipped


def test_unlabeled_traces_and_missing_metric_labels_are_ignored():
    traces = [_trace("a", 1)]
    result = align(traces, {"r:a": _label("a", "pass")})  # no per-metric labels
    assert next(a for a in result if a.metric == "faithfulness").n == 0
    assert "No overlap" not in render_alignment(result)  # 'overall' still has data
    assert "No overlap" in render_alignment(align(traces, {}))


def test_render_lists_disagreements_with_both_reasons():
    traces = [_trace("a", 1, rel=1)]
    text = render_alignment(align(traces, {"r:a": _label("a", "fail")}))
    assert "FP (too lenient)" in text and "fail because" in text


def test_label_flow_saves_labels_resumes_and_skips_errors(tmp_path):
    traces = [_trace("a", 1), _trace("b", 1), _trace("c", 1), _trace("d", 1, error="boom")]
    path = str(tmp_path / "labels.jsonl")

    answers = iter(["p", "looks good", "f", "no citations", "cite-missing", "q"])
    saved = label_traces(traces, path, input_fn=lambda _: next(answers), output_fn=lambda _: None)
    assert saved == 2
    labels = load_labels(path)
    assert labels["r:a"].verdict == "pass" and labels["r:b"].tag == "cite-missing"

    # resume: only "c" is left (a, b labeled; d errored)
    prompts: list[str] = []
    resumed = iter(["s"])
    label_traces(traces, path, input_fn=lambda p: (prompts.append(p), next(resumed))[1], output_fn=lambda _: None)
    assert len(prompts) == 1


def test_per_metric_labels_are_recorded(tmp_path):
    path = str(tmp_path / "labels.jsonl")
    answers = iter(["f", "bad", "", "y", "n", ""])  # verdict, reason, tag, then 3 metrics
    label_traces([_trace("a", 1)], path, per_metric=True, input_fn=lambda _: next(answers), output_fn=lambda _: None)
    assert load_labels(path)["r:a"].metrics == {"context_relevance": 1, "faithfulness": 0}
