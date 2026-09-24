"""Draft a test set from the tickets already in the vector store.

    python -m evals.generate_queries --n 30
    python -m evals.generate_queries --n 40 --unanswerable-fraction 0.25 --model gpt-4o-mini

This is a FIRST DRAFT -- you are the subject matter expert. Open
evals/datasets/queries.jsonl and delete, reword or add queries until it
reflects what your users really ask.

How queries are made (cross product, one LLM call per query so they don't all rhyme):
  - answerable / filtered: pick a real ticket (round-robin across components for
    coverage), pick a "shape" (pasted error, symptom, how-to, vague one-liner)
    and have the LLM write a *new* user report for that ticket's problem. The
    source ticket becomes `expected_ticket_ids`, which lets retrieval be scored
    without an LLM. Caveat: generated wording tends to echo the ticket, so
    hit@k on this set is optimistic -- reword some by hand.
  - unanswerable: a realistic support problem about a technology that is NOT
    Mesos; the right behaviour is to abstain.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.schemas import EvalQuery, write_jsonl  # noqa: E402

DEFAULT_OUTPUT = "evals/datasets/queries.jsonl"

SHAPES = {
    "error_paste": "Paste a realistic error message or log line the engineer is seeing, plus one sentence of context.",
    "symptom": "Describe the observed symptom in 2-3 sentences, as someone who has NOT diagnosed the cause.",
    "how_to": "Ask one direct 'why is X happening / how do I fix X' question.",
    "vague": "Write a short, vague one-line report, like a rushed chat message.",
}

OFF_TOPIC_TECH = [
    "PostgreSQL replication",
    "Terraform state locking",
    "Redis memory eviction",
    "Jenkins pipelines",
    "Nginx TLS configuration",
    "Elasticsearch shard allocation",
    "Kubernetes ingress",
    "Prometheus alert rules",
]

SYSTEM_PROMPT = (
    "You write realistic support requests that engineers would send to a support assistant. "
    "Output ONLY the text of the request -- no preamble, no quotes, no labels."
)

TICKET_RULES = """\
Write a NEW support request about the problem in the ticket below, as if the engineer just hit it \
and has not seen this ticket.
Style: {shape_instruction}
Rules:
- Do not mention the ticket ID and do not copy the ticket title's wording.
- Do not include the fix or the root cause.
- Do not use the words "ticket" or "Jira".

Ticket:
{ticket_text}"""

OFF_TOPIC_RULES = """\
Write a realistic support request about a problem with {tech}. It must be clearly about {tech} \
and must NOT mention Apache Mesos.
Style: {shape_instruction}"""


def primary_component(metadata: dict[str, Any]) -> str:
    return (metadata.get("component") or "unknown").split(";")[0].strip() or "unknown"


def select_seed_chunks(chunks: list[Any], n: int, rng: random.Random, min_chars: int = 200) -> list[Any]:
    """Pick up to n distinct tickets, round-robin across primary components for coverage.

    Uses each ticket's first chunk (title + problem statement); very short tickets
    are skipped because there's not enough to write a realistic report from.
    """
    groups: dict[str, list[Any]] = {}
    for chunk in chunks:
        if chunk.metadata.get("chunk_index") != 0 or not chunk.metadata.get("ticket_id"):
            continue
        if len(chunk.text) < min_chars:
            continue
        groups.setdefault(primary_component(chunk.metadata), []).append(chunk)

    for members in groups.values():
        members.sort(key=lambda c: c.metadata["ticket_id"])  # deterministic before shuffling
        rng.shuffle(members)
    order = sorted(groups)
    rng.shuffle(order)

    picked: list[Any] = []
    while len(picked) < n and any(groups.values()):
        for component in order:
            if groups[component] and len(picked) < n:
                picked.append(groups[component].pop())
    return picked


def _ask(llm: Any, prompt: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return text.strip().strip('"').strip()


def generate_queries(
    chunks: list[Any],
    llm: Any,
    n: int,
    *,
    unanswerable_fraction: float = 0.2,
    filtered_fraction: float = 0.15,
    seed: int = 42,
) -> list[EvalQuery]:
    rng = random.Random(seed)
    n_unanswerable = round(n * unanswerable_fraction)
    n_filtered = round(n * filtered_fraction)
    n_answerable = max(0, n - n_unanswerable - n_filtered)

    seeds = select_seed_chunks(chunks, n_answerable + n_filtered, rng)
    shapes = list(SHAPES)
    queries: list[EvalQuery] = []

    for index, chunk in enumerate(seeds):
        shape = shapes[index % len(shapes)]
        is_filtered = index >= n_answerable
        question = _ask(
            llm,
            TICKET_RULES.format(shape_instruction=SHAPES[shape], ticket_text=chunk.text[:1500]),
        )
        metadata = chunk.metadata
        queries.append(
            EvalQuery(
                id="",
                question=question,
                kind="filtered" if is_filtered else "answerable",
                expected_ticket_ids=[metadata["ticket_id"]],
                filters={"component": metadata["component"]} if is_filtered and metadata.get("component") else None,
                notes=f"generated from {metadata['ticket_id']} ({shape}): {metadata.get('summary', '')}",
            )
        )

    for index in range(n_unanswerable):
        tech = OFF_TOPIC_TECH[index % len(OFF_TOPIC_TECH)]
        shape = shapes[(index + 1) % len(shapes)]
        question = _ask(llm, OFF_TOPIC_RULES.format(tech=tech, shape_instruction=SHAPES[shape]))
        queries.append(EvalQuery(id="", question=question, kind="unanswerable", notes=f"off-topic: {tech} ({shape})"))

    rng.shuffle(queries)
    for number, query in enumerate(queries, start=1):
        query.id = f"q{number:03d}"
    return queries


def main() -> None:
    from langchain_openai import ChatOpenAI

    from app.config.settings import get_settings
    from app.vectorstore.chroma_store import VectorStoreService

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30, help="total queries to generate")
    parser.add_argument("--unanswerable-fraction", type=float, default=0.2)
    parser.add_argument("--filtered-fraction", type=float, default=0.15)
    parser.add_argument("--model", default=settings.judge_model)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing dataset (destroys hand edits)")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        sys.exit(f"{output} already exists and may contain your edits. Use --force to overwrite, or --output elsewhere.")

    chunks = VectorStoreService().get_chunks()
    if not chunks:
        sys.exit("The vector store is empty. Ingest data first (scripts/ingest.py).")

    llm = ChatOpenAI(model=args.model, api_key=settings.openai_api_key, temperature=0.8)
    queries = generate_queries(
        chunks,
        llm,
        args.n,
        unanswerable_fraction=args.unanswerable_fraction,
        filtered_fraction=args.filtered_fraction,
        seed=args.seed,
    )
    write_jsonl(output, queries)

    counts = {kind: sum(q.kind == kind for q in queries) for kind in ("answerable", "filtered", "unanswerable")}
    print(f"Wrote {len(queries)} queries to {output}  {counts}")
    print("Now review and edit the file by hand -- you are the subject matter expert.")


if __name__ == "__main__":
    main()
