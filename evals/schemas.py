"""Data shapes shared by every eval module, plus JSONL read/write helpers.

Everything the framework persists is one JSON object per line:
  - datasets/queries.jsonl      -> `EvalQuery`  (the test set; hand-editable)
  - results/<run_id>.jsonl      -> `Trace`      (one per query per run)
  - labels/human_labels.jsonl   -> `HumanLabel` (your pass/fail verdicts)
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

# "answerable": the source ticket should be retrievable and the answer grounded.
# "unanswerable": the corpus cannot answer it; the right behaviour is to abstain.
# "filtered": answerable, but asked with a metadata filter (exercises that path).
QUERY_KINDS = ("answerable", "unanswerable", "filtered")

T = TypeVar("T")


@dataclass
class EvalQuery:
    id: str
    question: str
    kind: str = "answerable"
    # Any one of these counts as a hit (some tickets have more than one equally
    # valid source, e.g. a duplicate-triage case citing the original ticket).
    expected_ticket_ids: list[str] = field(default_factory=list)
    filters: dict[str, Any] | None = None
    notes: str = ""
    # Free-text grouping from wherever the query came from (e.g. a human-authored
    # sheet's "Triage - Duplicate" / "Stale doc" / "Abstain"). Purely informational --
    # `kind` is what the checks/judges actually key off of.
    category: str = ""


@dataclass
class JudgeVerdict:
    """A binary verdict. `score` is None when the judge output couldn't be parsed."""

    score: int | None
    reason: str


@dataclass
class Trace:
    run_id: str
    query_id: str
    question: str
    kind: str
    chat_model: str
    top_k: int | None
    expected_ticket_ids: list[str] = field(default_factory=list)
    filters: dict[str, Any] | None = None
    category: str = ""
    answer: str = ""
    context: str = ""  # the evidence text exactly as the answering LLM saw it
    retrieved_ticket_ids: list[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    total_seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int | None = None  # stays None until the agentic flow exists
    error: str | None = None
    checks: dict[str, bool | None] = field(default_factory=dict)
    judgments: dict[str, JudgeVerdict] = field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        # Labels are keyed on this, so re-judging a run keeps human labels valid
        # (same answers), while re-running the app produces a new run_id.
        return f"{self.run_id}:{self.query_id}"

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass
class HumanLabel:
    """Your verdict on one trace.

    `verdict` is the overall pass/fail on the answer, `reason` a short note.
    `tag` is an optional short failure category you assign after seeing
    patterns. `metrics` optionally records your verdict per judged metric
    (e.g. {"faithfulness": 0}), which is what judge alignment compares against.
    """

    trace_id: str
    verdict: str  # "pass" | "fail"
    reason: str = ""
    tag: str = ""
    metrics: dict[str, int] = field(default_factory=dict)


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]


def trace_from_dict(data: dict[str, Any]) -> Trace:
    trace = _from_dict(Trace, data)
    trace.judgments = {name: _from_dict(JudgeVerdict, v) for name, v in (data.get("judgments") or {}).items()}
    return trace


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")


def append_jsonl(path: str | Path, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_queries(path: str | Path) -> list[EvalQuery]:
    return [_from_dict(EvalQuery, row) for row in read_jsonl(path)]


def load_traces(path: str | Path) -> list[Trace]:
    return [trace_from_dict(row) for row in read_jsonl(path)]


def load_labels(path: str | Path) -> dict[str, HumanLabel]:
    """Labels keyed by trace_id; a later line for the same trace overrides an earlier one."""
    return {row["trace_id"]: _from_dict(HumanLabel, row) for row in read_jsonl(path)}


QUERY_COLUMNS = ["id", "question", "kind", "expected_ticket_ids", "filters", "category", "notes"]

# `expected_ticket_ids` is a list in the data model but a single text cell in any table
# editor -- store/parse it the same "; "-joined way the source spreadsheet already used
# (e.g. "MESOS-1164; MESOS-1266"), so pasting from a sheet needs no reformatting.
_ID_SEPARATOR_RE = re.compile(r"[;,]\s*")


def _parse_ticket_ids(raw: str) -> list[str]:
    return [part.strip() for part in _ID_SEPARATOR_RE.split(raw.strip()) if part.strip()]


def queries_to_rows(queries: list[EvalQuery]) -> list[dict[str, Any]]:
    """Flat, editable rows (filters as a JSON string) for a table editor."""
    return [
        {
            "id": q.id,
            "question": q.question,
            "kind": q.kind,
            "expected_ticket_ids": "; ".join(q.expected_ticket_ids),
            "filters": json.dumps(q.filters) if q.filters else "",
            "category": q.category,
            "notes": q.notes,
        }
        for q in queries
    ]


def rows_to_queries(rows: list[dict[str, Any]]) -> list[EvalQuery]:
    """Inverse of `queries_to_rows` for hand-edited data.

    Blank-question rows are dropped, missing ids are assigned (q001, q002, ... skipping
    any already used), and `filters` must be empty or a JSON object -- otherwise ValueError.
    """
    rows = [r for r in rows if str(r.get("question") or "").strip()]
    used = {str(r["id"]).strip() for r in rows if str(r.get("id") or "").strip()}
    counter = 0
    queries = []
    for row in rows:
        query_id = str(row.get("id") or "").strip()
        if not query_id:
            counter += 1
            while f"q{counter:03d}" in used:
                counter += 1
            query_id = f"q{counter:03d}"
            used.add(query_id)
        raw_filters = str(row.get("filters") or "").strip()
        try:
            filters = json.loads(raw_filters) if raw_filters else None
        except json.JSONDecodeError as exc:
            raise ValueError(f"{query_id}: filters is not valid JSON ({exc})") from exc
        if filters is not None and not isinstance(filters, dict):
            raise ValueError(f'{query_id}: filters must be a JSON object like {{"component": "docker"}}')
        kind = str(row.get("kind") or "answerable")
        if kind not in QUERY_KINDS:
            raise ValueError(f"{query_id}: kind must be one of {QUERY_KINDS}, got '{kind}'")
        queries.append(
            EvalQuery(
                id=query_id,
                question=str(row["question"]).strip(),
                kind=kind,
                expected_ticket_ids=_parse_ticket_ids(str(row.get("expected_ticket_ids") or "")),
                filters=filters,
                notes=str(row.get("notes") or ""),
                category=str(row.get("category") or ""),
            )
        )
    ids = [q.id for q in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("query ids must be unique")
    return queries
