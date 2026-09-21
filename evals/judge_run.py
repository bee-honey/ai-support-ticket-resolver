"""Re-judge an existing run without re-running the app.

Use this in the alignment loop: edit a judge prompt in evals/judges/prompts/,
re-judge the same traces, then `python -m evals.align` to see whether the
judge now agrees with your labels better. Because the answers are unchanged,
your human labels for the run stay valid.

    python -m evals.judge_run evals/results/<run>.jsonl
    python -m evals.judge_run evals/results/<run>.jsonl --metrics faithfulness --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from evals.judges.base import METRICS, build_judges, judge_traces  # noqa: E402
from evals.schemas import load_traces, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="path to a results/<run>.jsonl file")
    parser.add_argument("--judge-model", default=get_settings().judge_model)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", help="write here instead of overwriting the run file")
    args = parser.parse_args()

    traces = load_traces(args.run)
    judge_traces(traces, build_judges(args.judge_model, tuple(args.metrics)), workers=args.workers)
    write_jsonl(args.output or args.run, traces)
    print(f"Re-judged {len(traces)} traces on {', '.join(args.metrics)} with {args.judge_model}")


if __name__ == "__main__":
    main()
