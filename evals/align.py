"""Compare each LLM judge with your human labels -- the "align the judges" step.

    python -m evals.align evals/results/<run>.jsonl

Positive = pass, following the course convention:
  false positive = judge says PASS, you said FAIL (judge too lenient)
  false negative = judge says FAIL, you said PASS (judge too strict)

Workflow: read the disagreements, edit the judge's prompt in
evals/judges/prompts/, `python -m evals.judge_run <run>` to re-judge the same
traces, and re-run this until agreement is acceptable. Only then trust the
judges on data you haven't labeled (synthetic queries, production samples).

The "overall" row compares your pass/fail verdict on the answer with the two
answer-level judges (faithfulness AND answer_relevancy). context_relevance is a
retrieval diagnostic, not an answer verdict, so it needs per-metric labels
(`evals.label --per-metric`) to be aligned.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.judges.base import METRICS  # noqa: E402
from evals.schemas import HumanLabel, Trace, load_labels, load_traces  # noqa: E402

DEFAULT_LABELS = "evals/labels/human_labels.jsonl"
ANSWER_METRICS = ("faithfulness", "answer_relevancy")


@dataclass
class Disagreement:
    trace: Trace
    label: HumanLabel
    human: int
    judge: int


@dataclass
class Alignment:
    metric: str
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def agreement(self) -> float | None:
        return (self.tp + self.tn) / self.n if self.n else None

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None


def human_value(label: HumanLabel, metric: str) -> int | None:
    if metric == "overall":
        return 1 if label.verdict == "pass" else 0
    return label.metrics.get(metric)


def judge_value(trace: Trace, metric: str) -> int | None:
    if metric == "overall":
        scores = [trace.judgments[m].score if m in trace.judgments else None for m in ANSWER_METRICS]
        if any(score is None for score in scores):
            return None
        return int(all(scores))
    verdict = trace.judgments.get(metric)
    return verdict.score if verdict else None


def align(traces: list[Trace], labels: dict[str, HumanLabel]) -> list[Alignment]:
    results = []
    for metric in (*METRICS, "overall"):
        alignment = Alignment(metric)
        for trace in traces:
            label = labels.get(trace.trace_id)
            if label is None:
                continue
            human, judge = human_value(label, metric), judge_value(trace, metric)
            if human is None or judge is None:
                continue
            if judge == 1 and human == 1:
                alignment.tp += 1
            elif judge == 0 and human == 0:
                alignment.tn += 1
            elif judge == 1 and human == 0:
                alignment.fp += 1
            else:
                alignment.fn += 1
            if human != judge:
                alignment.disagreements.append(Disagreement(trace, label, human, judge))
        results.append(alignment)
    return results


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(100 * value)}%"


def render_alignment(alignments: list[Alignment], max_disagreements: int = 10) -> str:
    lines = ["Judge vs human labels (positive = pass)", ""]
    lines.append(f"  {'metric':<20}{'n':>4}{'agree':>8}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'prec':>7}{'recall':>8}")
    for a in alignments:
        lines.append(
            f"  {a.metric:<20}{a.n:>4}{_pct(a.agreement):>8}{a.tp:>5}{a.tn:>5}{a.fp:>5}{a.fn:>5}"
            f"{_pct(a.precision):>7}{_pct(a.recall):>8}"
        )
    for a in alignments:
        if a.n == 0:
            continue
        human_fails = a.tn + a.fp
        lines.append(f"\n{a.metric}: judge caught {a.tn} of {human_fails} answers you failed")
        for d in a.disagreements[:max_disagreements]:
            kind = "FP (too lenient)" if d.judge == 1 else "FN (too strict)"
            verdict = d.trace.judgments.get(a.metric)
            judge_reason = verdict.reason if verdict else "; ".join(
                f"{m}: {v.reason}" for m, v in d.trace.judgments.items() if m in ANSWER_METRICS and v.score == 0
            ) or "all answer judges passed"
            lines.append(f"  {d.trace.query_id} {kind}")
            lines.append(f"    you:   {'pass' if d.human else 'fail'} - {d.label.reason or '(no reason)'}")
            lines.append(f"    judge: {'pass' if d.judge else 'fail'} - {judge_reason}")
        if len(a.disagreements) > max_disagreements:
            lines.append(f"  ... and {len(a.disagreements) - max_disagreements} more")
    if all(a.n == 0 for a in alignments):
        lines.append("\nNo overlap between labels and judged traces. Label some first: python -m evals.label <run>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="path to a results/<run>.jsonl file")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--max-disagreements", type=int, default=10)
    args = parser.parse_args()
    print(render_alignment(align(load_traces(args.run), load_labels(args.labels)), args.max_disagreements))


if __name__ == "__main__":
    main()
