"""Prompt templates for the RAG service, kept separate from business logic."""

from __future__ import annotations

from app.models.schemas import RetrievedChunk

SYSTEM_PROMPT = """\
You are a support engineering assistant. You help engineers resolve new \
support tickets by grounding your answer in retrieved historical tickets \
and support documentation.

Rules you must follow:
- Answer using ONLY the retrieved evidence provided below. Do not invent \
resolutions or steps that are not supported by the evidence.
- If the retrieved evidence is insufficient or only weakly related, say so \
explicitly instead of guessing. It is acceptable, and preferred, to answer \
"There is not enough supporting evidence to recommend a resolution."
- Mention the relevant historical ticket IDs (e.g. MESOS-1234) that support \
your answer.
- Structure your response in two clearly separated parts where practical:
  1. "Suggested Resolution" -- the concrete steps to try.
  2. "Supporting Evidence" -- which historical tickets/docs justify each step.
"""

USER_PROMPT_TEMPLATE = """\
New support problem:
{question}

Retrieved evidence:
{context}

Using only the retrieved evidence above, provide a grounded suggested \
resolution, citing ticket IDs where applicable.
"""


def format_chunk(chunk: RetrievedChunk, index: int) -> str:
    metadata = chunk.metadata
    tags = []
    if metadata.get("ticket_id"):
        tags.append(f"ticket_id={metadata['ticket_id']}")
    if metadata.get("component"):
        tags.append(f"component={metadata['component']}")
    if metadata.get("status"):
        tags.append(f"status={metadata['status']}")
    if metadata.get("source_file"):
        tags.append(f"source={metadata['source_file']}")
    header = f"[Evidence {index}] " + ", ".join(tags) if tags else f"[Evidence {index}]"
    return f"{header}\n{chunk.text}"


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(format_chunk(chunk, i + 1) for i, chunk in enumerate(chunks))


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return USER_PROMPT_TEMPLATE.format(question=question, context=build_context(chunks))
