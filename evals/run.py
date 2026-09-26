"""Run the test set through the RAG service, record traces, run checks + judges.

    python -m evals.run                                  # defaults from .env
    python -m evals.run --chat-model gpt-4o --k 8
    python -m evals.run --no-judge                       # checks only (free)
    python -m evals.judge_run evals/results/<run>.jsonl  # re-judge an existing run

Queries run one at a time on purpose: concurrency would distort the latency
numbers, and latency is one of the metrics being measured.

`run_eval` takes the service and models as arguments (no globals), so a UI can
call it with whatever model the user picked.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.rag.prompts import build_context  # noqa: E402
from evals.checks import DEFAULT_LATENCY_CAP_SECONDS, run_checks  # noqa: E402
from evals.judges.base import Judge, build_judges, judge_traces  # noqa: E402
from evals.schemas import EvalQuery, Trace, load_queries, write_jsonl  # noqa: E402

DEFAULT_DATASET = "evals/datasets/queries.jsonl"
DEFAULT_RESULTS_DIR = "evals/results"


def make_run_id(chat_model: str, dataset: str) -> str:
    """<timestamp>-<dataset>-<chat_model>, e.g. 20260925-100829-team_test_cases-gpt-4o-mini.

    Timestamp leads so `list_runs()`'s plain string sort stays newest-first; the dataset
    name is what was missing before -- with several test-set files now selectable (see
    the Run tab's dataset picker / `--dataset`), the run_id/filename is the only place
    that records which one produced a given run, so it needs to be legible at a glance
    rather than requiring someone to open the file and guess from its query_id prefixes.
    """
    dataset_name = Path(dataset).stem
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{dataset_name}-{chat_model}"


def run_query(
    query: EvalQuery,
    service: Any,
    *,
    run_id: str,
    chat_model: str,
    k: int | None,
    latency_cap_seconds: float = DEFAULT_LATENCY_CAP_SECONDS,
) -> Trace:
    trace = Trace(
        run_id=run_id,
        query_id=query.id,
        question=query.question,
        kind=query.kind,
        chat_model=chat_model,
        top_k=k,
        expected_ticket_ids=query.expected_ticket_ids,
        filters=query.filters,
        category=query.category,
    )
    try:
        result = service.answer(query.question, k=k, filters=query.filters)
    except Exception as exc:  # a failing query is data, not a reason to abort the run
        trace.error = f"{type(exc).__name__}: {exc}"
    else:
        trace.answer = result.answer
        trace.context = build_context(result.chunks) if result.chunks else ""
        trace.retrieved_ticket_ids = [c.ticket_id for c in result.chunks if c.ticket_id]
        trace.retrieval_seconds = result.retrieval_seconds
        trace.generation_seconds = result.generation_seconds
        trace.total_seconds = result.total_seconds
        trace.input_tokens = result.input_tokens
        trace.output_tokens = result.output_tokens
    trace.checks = run_checks(trace, latency_cap_seconds)
    return trace


def run_eval(
    queries: list[EvalQuery],
    service: Any,
    *,
    chat_model: str,
    k: int | None = None,
    run_id: str | None = None,
    latency_cap_seconds: float = DEFAULT_LATENCY_CAP_SECONDS,
    judges: dict[str, Judge] | None = None,
    workers: int = 4,
    on_progress: Callable[[int, int, Trace], None] | None = None,
) -> list[Trace]:
    # "adhoc": run_eval() takes already-loaded queries, not a dataset path, so it has no
    # real dataset name to offer here -- both real callers (the CLI and the Run tab)
    # already pass run_id explicitly with the actual dataset name; this fallback only
    # matters for a caller (e.g. a test) that doesn't precompute one.
    run_id = run_id or make_run_id(chat_model, "adhoc")
    traces: list[Trace] = []
    for index, query in enumerate(queries, start=1):
        trace = run_query(
            query, service, run_id=run_id, chat_model=chat_model, k=k, latency_cap_seconds=latency_cap_seconds
        )
        traces.append(trace)
        if on_progress:
            on_progress(index, len(queries), trace)
    if judges:
        judge_traces(traces, judges, workers=workers)
    return traces


def main() -> None:
    from app.rag.service import RAGService
    from evals.report import render_report

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--chat-model", default=settings.chat_model, help="model that answers the queries")
    parser.add_argument("--judge-model", default=settings.judge_model, help="model used by the LLM judges")
    parser.add_argument("--k", type=int, default=settings.default_top_k, help="chunks to retrieve per query")
    parser.add_argument("--latency-cap", type=float, default=DEFAULT_LATENCY_CAP_SECONDS)
    parser.add_argument("--workers", type=int, default=4, help="parallel judge calls")
    parser.add_argument("--limit", type=int, help="only run the first N queries")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM judges (deterministic checks only)")
    args = parser.parse_args()

    queries = load_queries(args.dataset)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        sys.exit(f"No queries found in {args.dataset}. Generate some: python -m evals.generate_queries")

    run_id = make_run_id(args.chat_model, args.dataset)
    output_path = Path(args.output_dir) / f"{run_id}.jsonl"
    print(f"Run {run_id}: {len(queries)} queries, chat model {args.chat_model}, k={args.k}")

    def progress(done: int, total: int, trace: Trace) -> None:
        status = f"ERROR {trace.error}" if trace.error else f"{trace.total_seconds:.1f}s"
        print(f"  [{done}/{total}] {trace.query_id}: {status}")

    service = RAGService(chat_model=args.chat_model)
    traces = run_eval(
        queries,
        service,
        chat_model=args.chat_model,
        k=args.k,
        run_id=run_id,
        latency_cap_seconds=args.latency_cap,
        on_progress=progress,
    )
    write_jsonl(output_path, traces)  # checkpoint before the (slower, fallible) judging step

    if not args.no_judge:
        print(f"Judging with {args.judge_model} ...")
        judge_traces(traces, build_judges(args.judge_model), workers=args.workers)
        write_jsonl(output_path, traces)

    print(f"\nSaved {output_path}\n")
    print(render_report(traces))


if __name__ == "__main__":
    main()
