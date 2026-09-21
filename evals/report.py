"""Summarise a run: pass rates, latency, tokens, and the reasons behind failures.

    python -m evals.report evals/results/<run>.jsonl
    python -m evals.report evals/results/<run>.jsonl --labels evals/labels/human_labels.jsonl
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.schemas import QUERY_KINDS, HumanLabel, Trace, load_labels, load_traces  # noqa: E402

DEFAULT_LABELS = "evals/labels/human_labels.jsonl"


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _rate(outcomes: list[bool]) -> str:
    if not outcomes:
        return "-"
    passed = sum(outcomes)
    return f"{passed}/{len(outcomes)} ({round(100 * passed / len(outcomes))}%)"


def _fmt(value: float | None, spec: str = ".1f") -> str:
    return "n/a" if value is None else format(value, spec)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def check_outcomes(traces: list[Trace], name: str) -> list[bool]:
    return [t.checks[name] for t in traces if t.checks.get(name) is not None]


def judge_outcomes(traces: list[Trace], metric: str) -> list[bool]:
    return [
        t.judgments[metric].score == 1
        for t in traces
        if metric in t.judgments and t.judgments[metric].score is not None
    ]


def latency_stats(traces: list[Trace]) -> dict[str, float | None]:
    """p50/p95/max of total latency plus p50 of each stage, over traces that didn't error."""
    ok = [t for t in traces if not t.error]
    totals = [t.total_seconds for t in ok]
    return {
        "p50": percentile(totals, 50),
        "p95": percentile(totals, 95),
        "max": max(totals, default=None),
        "retrieval_p50": percentile([t.retrieval_seconds for t in ok], 50),
        "generation_p50": percentile([t.generation_seconds for t in ok], 50),
    }


def token_stats(traces: list[Trace]) -> dict[str, float | None]:
    """Mean input/output tokens and tool calls per query (None when never reported)."""
    ok = [t for t in traces if not t.error]
    return {
        "input": _mean([t.input_tokens for t in ok if t.input_tokens is not None]),
        "output": _mean([t.output_tokens for t in ok if t.output_tokens is not None]),
        "tool_calls": _mean([t.tool_calls for t in ok if t.tool_calls is not None]),
    }


def metric_rows(traces: list[Trace]) -> list[dict]:
    """One row per (metric, kind): the data behind every pass-rate table in the UI.

    `group` is "check" (deterministic) or "judge" (LLM); `kind` is "all" or a query kind.
    """
    kinds = ["all", *[k for k in QUERY_KINDS if any(t.kind == k for t in traces)]]
    names: list[tuple[str, str, Callable[[list[Trace]], list[bool]]]] = []
    for name in list(traces[0].checks) if traces else []:
        names.append(("check", name, lambda ts, n=name: check_outcomes(ts, n)))
    for metric in sorted({m for t in traces for m in t.judgments}):
        names.append(("judge", metric, lambda ts, m=metric: judge_outcomes(ts, m)))

    rows = []
    for group, name, outcomes_for in names:
        for kind in kinds:
            outcomes = outcomes_for(traces if kind == "all" else [t for t in traces if t.kind == kind])
            rows.append(
                {
                    "group": group,
                    "metric": name,
                    "kind": kind,
                    "passed": sum(outcomes),
                    "total": len(outcomes),
                    "rate": sum(outcomes) / len(outcomes) if outcomes else None,
                }
            )
    return rows


def list_runs(results_dir: str | Path = "evals/results") -> list[Path]:
    """Run files, newest first (run ids start with a timestamp)."""
    directory = Path(results_dir)
    return sorted(directory.glob("*.jsonl"), reverse=True) if directory.exists() else []


def _table(title: str, rows: dict[str, Callable[[list[Trace]], list[bool]]], traces: list[Trace]) -> list[str]:
    kinds = [k for k in QUERY_KINDS if any(t.kind == k for t in traces)]
    columns = ["all", *kinds]
    lines = [title, "  " + f"{'':<24}" + "".join(f"{c:<18}" for c in columns)]
    for name, outcomes_for in rows.items():
        cells = [_rate(outcomes_for(traces))] + [_rate(outcomes_for([t for t in traces if t.kind == k])) for k in kinds]
        lines.append("  " + f"{name:<24}" + "".join(f"{cell:<18}" for cell in cells))
    return lines


def render_report(traces: list[Trace], labels: dict[str, HumanLabel] | None = None, max_reasons: int = 5) -> str:
    if not traces:
        return "No traces."
    ok = [t for t in traces if not t.error]
    errors = len(traces) - len(ok)
    header = f"Run {traces[0].run_id} · chat model {traces[0].chat_model} · k={traces[0].top_k} · {len(traces)} traces"
    lines = [header + (f" · {errors} ERROR(S)" if errors else ""), ""]

    check_names = list(traces[0].checks)
    lines += _table("Deterministic checks", {n: (lambda ts, n=n: check_outcomes(ts, n)) for n in check_names}, traces)

    metrics = sorted({m for t in traces for m in t.judgments})
    if metrics:
        lines += ["", *_table("LLM judges (1 = pass)", {m: (lambda ts, m=m: judge_outcomes(ts, m)) for m in metrics}, traces)]
        unparsed = sum(1 for t in traces for v in t.judgments.values() if v.score is None)
        if unparsed:
            lines.append(f"  ! {unparsed} judge verdict(s) could not be parsed / errored")

    latency, tokens = latency_stats(traces), token_stats(traces)
    lines += [
        "",
        "Latency (seconds)",
        f"  total      p50 {_fmt(latency['p50'])}  p95 {_fmt(latency['p95'])}  max {_fmt(latency['max'])}",
        f"  retrieval  p50 {_fmt(latency['retrieval_p50'], '.2f')}"
        f"   generation  p50 {_fmt(latency['generation_p50'], '.2f')}",
        "",
        "Tokens / tool calls (per query, mean)",
        f"  in {_fmt(tokens['input'], '.0f')}  out {_fmt(tokens['output'], '.0f')}"
        f"  tool calls {_fmt(tokens['tool_calls'])}",
    ]

    failure_lines: list[str] = []
    for metric in metrics:
        failing = [t for t in traces if metric in t.judgments and t.judgments[metric].score == 0]
        if failing:
            failure_lines.append(f"  {metric} - {len(failing)} failing:")
            failure_lines += [f"    {t.query_id}: {t.judgments[metric].reason}" for t in failing[:max_reasons]]
            if len(failing) > max_reasons:
                failure_lines.append(f"    ... and {len(failing) - max_reasons} more")
    if failure_lines:
        lines += ["", "Judge failure reasons (group these by hand into failure modes)", *failure_lines]

    if labels:
        mine = [labels[t.trace_id] for t in traces if t.trace_id in labels]
        if mine:
            verdicts = Counter(label.verdict for label in mine)
            lines += ["", f"Human labels: {len(mine)} of {len(traces)} traces - {verdicts['pass']} pass, {verdicts['fail']} fail"]
            tags = Counter(label.tag for label in mine if label.verdict == "fail" and label.tag)
            lines += [f"  {tag}: {count}" for tag, count in tags.most_common()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="path to a results/<run>.jsonl file")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--max-reasons", type=int, default=5)
    args = parser.parse_args()
    print(render_report(load_traces(args.run), load_labels(args.labels), args.max_reasons))


if __name__ == "__main__":
    main()
