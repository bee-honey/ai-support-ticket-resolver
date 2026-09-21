"""LLM-as-a-judge: one narrow, binary judge per metric.

Each judge does exactly one thing (its system prompt lives in
`prompts/<metric>.md`), sees only the inputs that metric needs, and returns
{"score": 0|1, "reason": "..."}. There are deliberately no 1-5 or percentage
scores: LLMs cluster around the middle of a scale, while a binary verdict
forces a decision and makes the failing traces easy to group by reason.

The judge model is a parameter everywhere (default: `JUDGE_MODEL` setting), so a
UI can offer a model picker without touching this module.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config.settings import get_settings
from evals.checks import is_abstention
from evals.schemas import JudgeVerdict, Trace

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Which trace fields each metric is allowed to see (metrics stay independent).
METRIC_INPUTS: dict[str, tuple[str, ...]] = {
    "context_relevance": ("question", "context"),
    "faithfulness": ("question", "context", "answer"),
    "answer_relevancy": ("question", "answer"),
}
METRICS = tuple(METRIC_INPUTS)

_INPUT_LABELS = {"question": "PROBLEM", "context": "RETRIEVED EVIDENCE", "answer": "ANSWER"}


def load_system_prompt(metric: str) -> str:
    return (PROMPTS_DIR / f"{metric}.md").read_text(encoding="utf-8").strip()


def build_user_message(metric: str, trace: Trace) -> str:
    values = {"question": trace.question, "context": trace.context, "answer": trace.answer}
    return "\n\n".join(f"### {_INPUT_LABELS[name]}\n{values[name]}" for name in METRIC_INPUTS[metric])


def parse_verdict(text: str) -> JudgeVerdict:
    """Parse a judge reply; tolerate stray prose or code fences around the JSON."""
    candidate = text.strip()
    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        return JudgeVerdict(score=None, reason=f"unparseable judge output: {candidate[:80]}")

    raw_score = data.get("score")
    score = {0: 0, 1: 1, "0": 0, "1": 1, False: 0, True: 1}.get(raw_score) if isinstance(raw_score, (int, str, bool)) else None
    return JudgeVerdict(score=score, reason=str(data.get("reason", "")).strip())


class Judge:
    def __init__(self, metric: str, model: str | None = None, llm: Any | None = None):
        if metric not in METRIC_INPUTS:
            raise ValueError(f"unknown metric '{metric}'; expected one of {METRICS}")
        settings = get_settings()
        self.metric = metric
        self.model = model or settings.judge_model
        self._system_prompt = load_system_prompt(metric)
        self._llm = llm or ChatOpenAI(
            model=self.model,
            api_key=settings.openai_api_key,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def judge(self, trace: Trace) -> JudgeVerdict:
        # With nothing retrieved there is nothing for the LLM to weigh; decide directly.
        if not trace.context:
            if self.metric == "context_relevance":
                return JudgeVerdict(score=0, reason="nothing was retrieved")
            if self.metric == "faithfulness":
                ok = is_abstention(trace.answer)
                return JudgeVerdict(score=int(ok), reason="abstained" if ok else "answered with no evidence")

        messages = [SystemMessage(content=self._system_prompt), HumanMessage(content=build_user_message(self.metric, trace))]
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:  # network/rate-limit: record it, don't abort the whole run
            return JudgeVerdict(score=None, reason=f"judge error: {exc}")
        content = response.content if isinstance(response.content, str) else str(response.content)
        return parse_verdict(content)


def build_judges(model: str | None = None, metrics: tuple[str, ...] = METRICS) -> dict[str, Judge]:
    return {metric: Judge(metric, model=model) for metric in metrics}


def judge_traces(traces: list[Trace], judges: dict[str, Judge], workers: int = 4) -> None:
    """Fill `trace.judgments` in place. Judging is parallel; traces with errors are skipped."""
    tasks = [(trace, name) for trace in traces if not trace.error for name in judges]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        verdicts = list(pool.map(lambda task: judges[task[1]].judge(task[0]), tasks))
    for (trace, name), verdict in zip(tasks, verdicts):
        trace.judgments[name] = verdict
