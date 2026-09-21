"""Terminal tool for labeling traces pass/fail -- your ground truth for aligning the judges.

    python -m evals.label evals/results/<run>.jsonl
    python -m evals.label evals/results/<run>.jsonl --per-metric   # also label each judged metric
    python -m evals.label evals/results/<run>.jsonl --relabel

Judge verdicts are deliberately hidden while you label so they can't bias you.
Give a pass/fail on the ANSWER plus a short reason ("no citations",
"wrong component", ...). Once you see repeated reasons, reuse one as the `tag`
so failures can be counted and the biggest problem tackled first.
Labels are saved after every trace, so you can quit (q) and resume later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.judges.base import METRICS  # noqa: E402
from evals.schemas import HumanLabel, Trace, append_jsonl, load_labels, load_traces  # noqa: E402

DEFAULT_LABELS = "evals/labels/human_labels.jsonl"
CONTEXT_PREVIEW_CHARS = 350


def _preview_context(context: str) -> str:
    blocks = context.split("\n\n[Evidence ")
    shown = []
    for i, block in enumerate(blocks):
        block = block if i == 0 else "[Evidence " + block
        shown.append(block if len(block) <= CONTEXT_PREVIEW_CHARS else block[:CONTEXT_PREVIEW_CHARS] + " ...")
    return "\n".join(shown)


def label_traces(
    traces: list[Trace],
    labels_path: str,
    *,
    relabel: bool = False,
    per_metric: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Interactively label traces; returns the number of labels saved."""
    existing = load_labels(labels_path)
    todo = [t for t in traces if not t.error and (relabel or t.trace_id not in existing)]
    output_fn(f"{len(todo)} trace(s) to label ({len(existing)} already labeled). p=pass f=fail s=skip q=quit\n")

    saved = 0
    for position, trace in enumerate(todo, start=1):
        output_fn(f"{'=' * 78}\n[{position}/{len(todo)}] {trace.query_id}  ({trace.kind})  {trace.total_seconds:.1f}s")
        output_fn(f"\nPROBLEM:\n{trace.question}")
        output_fn(f"\nEVIDENCE SHOWN:\n{_preview_context(trace.context) or '(none retrieved)'}")
        output_fn(f"\nANSWER:\n{trace.answer}\n")

        choice = ""
        while choice not in ("p", "f", "s", "q"):
            choice = input_fn("verdict [p/f/s/q]: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue

        reason = input_fn("reason (short): ").strip()
        tag = input_fn("tag (optional failure category): ").strip() if choice == "f" else ""
        metrics: dict[str, int] = {}
        if per_metric:
            for metric in METRICS:
                answer = input_fn(f"  {metric} ok? [y/n/enter=skip]: ").strip().lower()
                if answer in ("y", "n"):
                    metrics[metric] = 1 if answer == "y" else 0

        append_jsonl(
            labels_path,
            HumanLabel(
                trace_id=trace.trace_id,
                verdict="pass" if choice == "p" else "fail",
                reason=reason,
                tag=tag,
                metrics=metrics,
            ),
        )
        saved += 1
    output_fn(f"\nSaved {saved} label(s) to {labels_path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="path to a results/<run>.jsonl file")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--per-metric", action="store_true")
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()
    label_traces(load_traces(args.run), args.labels, relabel=args.relabel, per_metric=args.per_metric)


if __name__ == "__main__":
    main()
