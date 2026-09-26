"""Evals page: run the test set, read the metrics, label answers, align the judges.

The UI holds no eval logic of its own -- it calls the same `evals/` functions the
CLI does (`run_eval`, `judge_traces`, `align`, ...), so the terminal workflow and
this page always agree. Models are chosen here per run (chat model + judge model).

Status is always shown as icon + word (never colour alone), and pass rates as
single-hue bars with the number beside them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import get_settings, model_choices  # noqa: E402
from app.rag.service import RAGService  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from evals.align import align  # noqa: E402
from evals.checks import DEFAULT_LATENCY_CAP_SECONDS  # noqa: E402
from evals.judges.base import METRICS, PROMPTS_DIR, build_judges, judge_traces, load_system_prompt  # noqa: E402
from evals.report import latency_stats, list_runs, metric_rows, token_stats  # noqa: E402
from evals.run import make_run_id, run_eval  # noqa: E402
from evals.schemas import (  # noqa: E402
    QUERY_COLUMNS,
    QUERY_KINDS,
    HumanLabel,
    Trace,
    append_jsonl,
    list_datasets,
    load_labels,
    load_queries,
    load_traces,
    queries_to_rows,
    rows_to_queries,
    write_jsonl,
)
from ui.ticket_links import ticket_page_link  # noqa: E402

# EVALS_DIR lets tests point the page at a temp folder instead of the real data.
EVALS_DIR = Path(os.getenv("EVALS_DIR") or ROOT / "evals")
DATASETS_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"
LABELS_PATH = EVALS_DIR / "labels/human_labels.jsonl"

METRIC_HELP = {
    "retrieval_hit": "The ticket the query was generated from was among the retrieved chunks.",
    "citations_valid": "Every ticket ID cited in the answer was actually retrieved (no invented citations).",
    "abstention_correct": "Off-topic queries were declined; answerable ones were not.",
    "has_required_sections": "Answer has 'Suggested Resolution' and 'Supporting Evidence' sections.",
    "within_latency": "Total response time was within the latency cap.",
    "context_relevance": "Judge: the retrieved evidence is useful for the problem.",
    "faithfulness": "Judge: every claim and cited ticket in the answer is supported by the evidence.",
    "answer_relevancy": "Judge: the answer addresses the problem that was asked.",
}

@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_rag_service(chat_model: str) -> RAGService:
    return RAGService(retriever=get_retriever(), chat_model=chat_model)


def status_text(value: bool | int | None) -> str:
    if value is None:
        return "– n/a"
    return "✅ pass" if value else "❌ fail"


def pct(rate: float | None) -> float | None:
    return None if rate is None else round(100 * rate, 1)


def model_picker(label: str, default: str, key: str) -> str:
    choices = model_choices(default)
    return st.selectbox(label, choices, index=choices.index(default), key=key)


# --------------------------------------------------------------------------- run tab


def render_run_tab() -> None:
    settings = get_settings()

    dataset_files = list_datasets(DATASETS_DIR)
    if not dataset_files:
        st.info(
            "No test set files in `evals/datasets/` yet. Generate a draft:\n\n"
            "`python -m evals.generate_queries --n 30`"
        )
        return

    names = [p.name for p in dataset_files]
    default_index = names.index("queries.jsonl") if "queries.jsonl" in names else 0
    dataset_name = st.selectbox(
        "Test set",
        names,
        index=names.index(st.session_state["selected_dataset"]) if st.session_state.get("selected_dataset") in names else default_index,
        key="selected_dataset",
        help="Files in evals/datasets/. Switch here to run or edit a different test set.",
    )
    dataset_path = DATASETS_DIR / dataset_name
    queries = load_queries(dataset_path)

    with st.expander(f"{dataset_name} — {len(queries)} queries (editable)", expanded=not queries):
        if not queries:
            st.info(
                "This test set is empty. Generate a draft, then edit it here:\n\n"
                "`python -m evals.generate_queries --n 30`\n\n"
                "or add rows below and save."
            )
        edited = st.data_editor(
            pd.DataFrame(queries_to_rows(queries), columns=QUERY_COLUMNS),
            num_rows="dynamic",
            width="stretch",
            key=f"query_editor_{dataset_name}",  # per-dataset key: switching files never leaks unsaved edits between them
            column_config={
                "kind": st.column_config.SelectboxColumn("kind", options=list(QUERY_KINDS), required=True),
                "question": st.column_config.TextColumn("question", width="large"),
                "filters": st.column_config.TextColumn("filters (JSON)", help='e.g. {"component": "docker"}'),
                "expected_ticket_ids": st.column_config.TextColumn(
                    "expected ticket(s)", help='Any one counts as a hit, e.g. "MESOS-1164; MESOS-1266"'
                ),
            },
        )
        if st.button("Save test set"):
            try:
                to_save = rows_to_queries(edited.fillna("").to_dict("records"))
                write_jsonl(dataset_path, to_save)
            except ValueError as exc:
                st.error(f"Not saved: {exc}")
            else:
                st.session_state["_flash"] = f"Saved {len(to_save)} queries to {dataset_name}."
                st.rerun()

    st.subheader("Configure a run")
    left, right = st.columns(2)
    with left:
        chat_model = model_picker("Chat model (answers the queries)", settings.chat_model, "run_chat_model")
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, settings.default_top_k, key="run_k")
        limit = st.number_input("Only run the first N queries (0 = all)", 0, max(len(queries), 1), 0, key="run_limit")
    with right:
        use_judges = st.checkbox("Run LLM judges", value=True, key="run_judges")
        judge_model = model_picker("Judge model", settings.judge_model, "run_judge_model")
        latency_cap = st.number_input(
            "Latency cap (seconds)", 1.0, 120.0, DEFAULT_LATENCY_CAP_SECONDS, step=1.0, key="run_cap"
        )

    if not settings.openai_api_key or settings.openai_api_key == "sk-changeme":
        st.warning("OPENAI_API_KEY is not set in `.env`; every query will error.", icon="⚠️")

    if st.button("▶ Run evals", type="primary", disabled=not queries):
        subset = queries[: int(limit)] if limit else queries
        run_id = make_run_id(chat_model, dataset_name)
        path = RESULTS_DIR / f"{run_id}.jsonl"
        bar = st.progress(0.0, text="Starting…")

        def on_progress(done: int, total: int, trace: Trace) -> None:
            bar.progress(done / total, text=f"Answering {done}/{total} · {trace.query_id}")

        traces = run_eval(
            subset,
            get_rag_service(chat_model),
            chat_model=chat_model,
            k=top_k,
            run_id=run_id,
            latency_cap_seconds=latency_cap,
            on_progress=on_progress,
        )
        write_jsonl(path, traces)  # checkpoint before judging
        if use_judges:
            bar.progress(1.0, text=f"Judging with {judge_model}…")
            judge_traces(traces, build_judges(judge_model))
            write_jsonl(path, traces)
        st.session_state["_select_run"] = path.name
        st.session_state["_flash"] = f"Run {run_id} finished ({len(traces)} queries)."
        st.rerun()


# ------------------------------------------------------------------------ metrics tab


def _rate_tile(column, label: str, rows: list[dict], previous: list[dict] | None) -> None:
    row = next((r for r in rows if r["metric"] == label and r["kind"] == "all"), None)
    if row is None:
        return
    rate = row["rate"]
    delta = None
    if previous is not None and rate is not None:
        prev = next((r for r in previous if r["metric"] == label and r["kind"] == "all"), None)
        if prev and prev["rate"] is not None:
            delta = f"{(rate - prev['rate']) * 100:+.0f} pts"
    column.metric(
        label.replace("_", " "),
        "–" if rate is None else f"{rate:.0%}",
        delta=delta,
        help=f"{METRIC_HELP.get(label, '')}  ({row['passed']}/{row['total']} passed)",
    )


def _pass_rate_frame(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table["rate"] = table["rate"].map(pct)
    wide = table.pivot(index=["group", "metric"], columns="kind", values="rate")
    counts = table[table["kind"] == "all"].set_index(["group", "metric"])
    wide.insert(0, "passed", counts["passed"].astype(str) + "/" + counts["total"].astype(str))
    ordered_kinds = ["all", *[k for k in QUERY_KINDS if k in wide.columns]]
    wide = wide[["passed", *ordered_kinds]].reset_index()
    wide["group"] = wide["group"].map({"check": "Deterministic check", "judge": "LLM judge"})
    return wide.sort_values(["group", "metric"], ascending=[False, True]).rename(columns={"group": "type"})


def _trace_frame(traces: list[Trace], labels: dict[str, HumanLabel]) -> pd.DataFrame:
    judge_names = sorted({m for t in traces for m in t.judgments})
    rows = []
    for t in traces:
        row = {"id": t.query_id, "kind": t.kind, "time (s)": round(t.total_seconds, 1)}
        row.update({name: status_text(value) for name, value in t.checks.items()})
        row.update({m: status_text(t.judgments[m].score if m in t.judgments else None) for m in judge_names})
        row["you"] = labels[t.trace_id].verdict if t.trace_id in labels else ""
        row["error"] = t.error or ""
        row["question"] = t.question
        rows.append(row)
    return pd.DataFrame(rows)


def _render_trace_detail(trace: Trace, labels: dict[str, HumanLabel]) -> None:
    st.markdown(f"#### {trace.query_id} · {trace.kind}")
    if trace.error:
        st.error(trace.error)
    st.markdown(f"**Problem**\n\n{trace.question}")
    st.markdown(f"**Answer**\n\n{trace.answer or '_(none)_'}")
    retrieved_ids = list(dict.fromkeys(trace.retrieved_ticket_ids))  # dedupe, preserve retrieval order
    if retrieved_ids:
        st.markdown("**Retrieved tickets** (click to verify against the real ticket)")
        for ticket_id in retrieved_ids:
            ticket_page_link(ticket_id)
    with st.expander("Evidence the model saw"):
        st.text(trace.context or "(nothing retrieved)")
    st.markdown("**Checks**")
    for name, value in trace.checks.items():
        st.markdown(f"- {status_text(value)} — `{name}`")
    if trace.judgments:
        st.markdown("**Judges**")
        for name, verdict in trace.judgments.items():
            st.markdown(f"- {status_text(verdict.score)} — `{name}`: {verdict.reason}")
    label = labels.get(trace.trace_id)
    if label:
        st.markdown(f"**Your label:** {label.verdict}" + (f" — {label.reason}" if label.reason else ""))


def render_metrics_tab(traces: list[Trace], previous: list[Trace] | None, labels: dict[str, HumanLabel]) -> None:
    rows = metric_rows(traces)
    prev_rows = metric_rows(previous) if previous else None
    lat, tok = latency_stats(traces), token_stats(traces)
    prev_lat = latency_stats(previous) if previous else None
    errors = sum(1 for t in traces if t.error)

    first = st.columns(5)
    first[0].metric("Queries", len(traces), delta=f"{errors} errored" if errors else None, delta_color="inverse")
    for column, key, label in ((first[1], "p50", "Latency p50"), (first[2], "p95", "Latency p95")):
        value = lat[key]
        delta = f"{value - prev_lat[key]:+.1f}s" if prev_lat and value is not None and prev_lat[key] is not None else None
        column.metric(
            label,
            "–" if value is None else f"{value:.1f}s",
            delta=delta,
            delta_color="inverse",
            help=f"retrieval p50 {lat['retrieval_p50'] or 0:.2f}s · generation p50 {lat['generation_p50'] or 0:.2f}s",
        )
    first[3].metric(
        "Tokens / query",
        "–" if tok["input"] is None else f"{(tok['input'] or 0) + (tok['output'] or 0):,.0f}",
        help=f"in {tok['input'] or 0:,.0f} · out {tok['output'] or 0:,.0f}",
    )
    first[4].metric("Tool calls / query", "n/a" if tok["tool_calls"] is None else f"{tok['tool_calls']:.1f}",
                    help="Stays n/a until the agentic flow records tool calls.")

    judge_metrics = [m for m in METRICS if any(r["metric"] == m for r in rows)]
    check_metrics = sorted({r["metric"] for r in rows if r["group"] == "check"})
    if judge_metrics:
        st.caption("LLM judges — pass rate")
        cols = st.columns(len(judge_metrics))
        for column, metric in zip(cols, judge_metrics):
            _rate_tile(column, metric, rows, prev_rows)
    st.caption("Deterministic checks — pass rate")
    cols = st.columns(max(len(check_metrics), 1))
    for column, metric in zip(cols, check_metrics):
        _rate_tile(column, metric, rows, prev_rows)
    if previous:
        st.caption("Arrows show the change versus the comparison run.")

    st.markdown("##### Pass rate by query kind")
    frame = _pass_rate_frame(rows)
    percent = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={kind: percent for kind in ("all", *QUERY_KINDS) if kind in frame.columns},
    )

    st.markdown("##### Traces")
    trace_frame = _trace_frame(traces, labels)
    controls = st.columns([2, 1])
    kinds = controls[0].multiselect("Kind", [k for k in QUERY_KINDS if k in trace_frame["kind"].unique()], key="trace_kinds")
    only_failing = controls[1].checkbox("Only failing", key="trace_failing")
    visible = trace_frame
    if kinds:
        visible = visible[visible["kind"].isin(kinds)]
    if only_failing:
        visible = visible[visible.apply(lambda r: any(str(v).startswith("❌") for v in r) or bool(r["error"]), axis=1)]
    st.caption(f"{len(visible)} of {len(trace_frame)} traces — click a row to inspect it")
    event = st.dataframe(
        visible, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="trace_table"
    )
    selected = event.selection.rows
    if selected:
        chosen_id = visible.iloc[selected[0]]["id"]
        _render_trace_detail(next(t for t in traces if t.query_id == chosen_id), labels)

    failing = {
        m: [t for t in traces if m in t.judgments and t.judgments[m].score == 0] for m in judge_metrics
    }
    if any(failing.values()):
        with st.expander("Why the judges failed answers (group these into failure modes)"):
            for metric, items in failing.items():
                if items:
                    st.markdown(f"**{metric}** — {len(items)} failing")
                    for t in items:
                        st.markdown(f"- `{t.query_id}` {t.judgments[metric].reason}")


# -------------------------------------------------------------------------- label tab


def render_label_tab(traces: list[Trace], labels: dict[str, HumanLabel]) -> None:
    labelable = [t for t in traces if not t.error]
    if not labelable:
        st.info("This run has no answered queries to label.")
        return
    done = sum(1 for t in labelable if t.trace_id in labels)
    st.progress(done / len(labelable), text=f"{done} of {len(labelable)} labeled")
    st.caption(
        "Judge verdicts are hidden on this tab so they don't bias you. Pass if the answer addresses the "
        "problem, its steps come from the evidence shown, and it cites real tickets from that evidence."
    )

    ids = [t.trace_id for t in labelable]
    next_unlabeled = next((t.trace_id for t in labelable if t.trace_id not in labels), ids[0])
    if (pending := st.session_state.pop("_label_next", None)) in ids:
        st.session_state["label_trace"] = pending
    if st.session_state.get("label_trace") not in ids:
        st.session_state["label_trace"] = next_unlabeled

    by_id = {t.trace_id: t for t in labelable}
    trace_id = st.selectbox(
        "Trace",
        ids,
        key="label_trace",
        format_func=lambda tid: f"{by_id[tid].query_id} · {by_id[tid].kind}"
        + (f"  (labeled: {labels[tid].verdict})" if tid in labels else ""),
    )
    trace = by_id[trace_id]

    st.markdown(f"**Problem**\n\n{trace.question}")
    st.markdown(f"**Answer**\n\n{trace.answer}")
    with st.expander("Evidence the model saw"):
        st.text(trace.context or "(nothing retrieved)")

    existing = labels.get(trace_id)
    with st.form(f"label_form_{trace_id}"):
        verdict = st.radio(
            "Verdict",
            ["pass", "fail"],
            index=1 if existing and existing.verdict == "fail" else 0,
            horizontal=True,
            format_func=lambda v: "✅ pass" if v == "pass" else "❌ fail",
        )
        reason = st.text_input("Reason (short)", value=existing.reason if existing else "")
        known_tags = sorted({l.tag for l in labels.values() if l.tag})
        tag = st.text_input(
            "Failure tag (optional; reuse a tag when you see the same problem again)",
            value=existing.tag if existing else "",
            help="Existing tags: " + (", ".join(known_tags) or "none yet"),
        )
        with st.expander("Per-metric labels (needed to align context_relevance)"):
            metric_answers = {
                m: st.selectbox(
                    m.replace("_", " "),
                    ["skip", "ok", "not ok"],
                    index=0 if not existing or m not in existing.metrics else (1 if existing.metrics[m] else 2),
                    key=f"pm_{trace_id}_{m}",
                    help=METRIC_HELP[m],
                )
                for m in METRICS
            }
        save = st.form_submit_button("Save & next", type="primary")

    if save:
        append_jsonl(
            LABELS_PATH,
            HumanLabel(
                trace_id=trace_id,
                verdict=verdict,
                reason=reason.strip(),
                tag=tag.strip() if verdict == "fail" else "",
                metrics={m: (1 if a == "ok" else 0) for m, a in metric_answers.items() if a != "skip"},
            ),
        )
        position = ids.index(trace_id)
        following = ids[position + 1 :] + ids[:position]
        target = next((tid for tid in following if tid not in labels), None)
        if target:
            st.session_state["_label_next"] = target
        st.rerun()


# -------------------------------------------------------------------------- align tab


def render_align_tab(run_path: Path, traces: list[Trace], labels: dict[str, HumanLabel]) -> None:
    st.caption(
        "Positive = pass. A false positive means the judge passed an answer you failed (too lenient); "
        "a false negative means it failed one you passed (too strict). Tune the prompt until they agree."
    )
    alignments = align(traces, labels)
    if all(a.n == 0 for a in alignments):
        st.info("No overlap yet between your labels and this run. Label some answers on the Label tab first.")
    else:
        frame = pd.DataFrame(
            [
                {
                    "metric": a.metric,
                    "labeled": a.n,
                    "agreement": pct(a.agreement),
                    "TP": a.tp, "TN": a.tn, "FP": a.fp, "FN": a.fn,
                    "precision": pct(a.precision),
                    "recall": pct(a.recall),
                }
                for a in alignments
            ]
        )
        percent = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")
        st.dataframe(
            frame, hide_index=True, width="stretch",
            column_config={"agreement": percent, "precision": percent, "recall": percent},
        )
        st.caption(
            "`overall` compares your pass/fail verdict with faithfulness AND answer_relevancy. "
            "`context_relevance` only appears once you fill in per-metric labels."
        )
        for a in alignments:
            if not a.disagreements:
                continue
            with st.expander(f"{a.metric}: {len(a.disagreements)} disagreement(s)"):
                for d in a.disagreements:
                    kind = "❌ judge too lenient (FP)" if d.judge == 1 else "❌ judge too strict (FN)"
                    verdict = d.trace.judgments.get(a.metric)
                    judge_reason = verdict.reason if verdict else "; ".join(
                        f"{m}: {v.reason}" for m, v in d.trace.judgments.items() if v.score == 0
                    ) or "all answer judges passed"
                    st.markdown(
                        f"**{d.trace.query_id}** — {kind}\n\n"
                        f"- you: {'pass' if d.human else 'fail'} — {d.label.reason or '(no reason)'}\n"
                        f"- judge: {'pass' if d.judge else 'fail'} — {judge_reason}"
                    )

    st.markdown("##### Tune a judge")
    settings = get_settings()
    metric = st.selectbox("Judge", METRICS, key="tune_metric")
    prompt_key = f"tune_prompt_{metric}"
    text = st.text_area("System prompt", load_system_prompt(metric), height=320, key=prompt_key)
    buttons = st.columns(3)
    judge_model = model_picker("Judge model", settings.judge_model, "tune_judge_model")
    if buttons[0].button("Save prompt"):
        (PROMPTS_DIR / f"{metric}.md").write_text(text.strip() + "\n", encoding="utf-8")
        st.success(f"Saved {metric}.md")
    if buttons[1].button("Save & re-judge this run"):
        (PROMPTS_DIR / f"{metric}.md").write_text(text.strip() + "\n", encoding="utf-8")
        with st.spinner(f"Re-judging {len(traces)} traces on {metric}…"):
            judge_traces(traces, build_judges(judge_model, (metric,)))
            write_jsonl(run_path, traces)
        st.session_state["_flash"] = f"Re-judged {metric} with {judge_model}; the run file was updated."
        st.rerun()
    st.caption("Re-judging overwrites that metric's verdicts in the selected run file. Your labels stay valid.")


# ------------------------------------------------------------------------------ page

st.caption("Measure answer quality, retrieval and latency on a test set, and check that the LLM judges agree with you.")

if flash := st.session_state.pop("_flash", None):
    st.success(flash)

run_files = list_runs(RESULTS_DIR)
run_names = [p.name for p in run_files]
if pending_run := st.session_state.pop("_select_run", None):
    st.session_state["selected_run"] = pending_run
if st.session_state.get("selected_run") not in run_names:
    st.session_state.pop("selected_run", None)

with st.sidebar:
    st.header("Run")
    if run_names:
        selected_name = st.selectbox("Selected run", run_names, key="selected_run")
        compare_name = st.selectbox("Compare with", ["(none)", *[n for n in run_names if n != selected_name]])
    else:
        selected_name, compare_name = None, "(none)"
        st.caption("No runs yet — start one on the Run tab.")

tab_run, tab_metrics, tab_label, tab_align = st.tabs(["▶ Run", "📈 Metrics", "🏷 Label", "🎯 Align"])

with tab_run:
    render_run_tab()

if selected_name:
    run_path = RESULTS_DIR / selected_name
    run_traces = load_traces(run_path)
    human_labels = load_labels(LABELS_PATH)
    compare_traces = load_traces(RESULTS_DIR / compare_name) if compare_name != "(none)" else None
    first = run_traces[0] if run_traces else None
    if first:
        st.sidebar.caption(f"Chat model: {first.chat_model} · k={first.top_k} · {len(run_traces)} traces")
    with tab_metrics:
        render_metrics_tab(run_traces, compare_traces, human_labels) if run_traces else st.info("This run is empty.")
    with tab_label:
        render_label_tab(run_traces, human_labels)
    with tab_align:
        render_align_tab(run_path, run_traces, human_labels)
else:
    for tab in (tab_metrics, tab_label, tab_align):
        with tab:
            st.info("No runs yet. Start one on the Run tab.")
