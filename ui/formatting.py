"""Text formatting for the response-metrics line shown under each answer.

Kept free of Streamlit so it can be unit-tested. `format_stats` is the final
snapshot (exact numbers); `format_live_stats` is the in-progress version shown
while an answer is still being produced.
"""

from __future__ import annotations

from app.models.schemas import RAGResult

SEPARATOR = "  ·  "


def format_stats(result: RAGResult, model: str | None = None) -> str:
    """Final one-line snapshot: model, response time and exact token usage."""
    parts = [f"🤖 {model}"] if model else []
    parts.append(
        f"⏱ {result.total_seconds:.1f}s "
        f"(retrieval {result.retrieval_seconds:.1f}s · generation {result.generation_seconds:.1f}s)"
    )
    if result.total_tokens is not None:
        parts.append(
            f"🔢 {result.total_tokens:,} tokens "
            f"({result.input_tokens or 0:,} in · {result.output_tokens or 0:,} out)"
        )
    return SEPARATOR.join(parts)


def format_live_stats(
    model: str | None,
    phase: str,
    *,
    retrieval_seconds: float = 0.0,
    generation_seconds: float = 0.0,
    tokens_out: int = 0,
) -> str:
    """In-progress line. `phase` is "retrieving" or "generating".

    Input tokens are only reported by the API once the response finishes, so the
    live line shows output tokens so far (approximate, "~"); the final line has both.
    """
    parts = [f"🤖 {model}"] if model else []
    if phase == "retrieving":
        parts.append("⏱ retrieving evidence…")
    else:
        total = retrieval_seconds + generation_seconds
        parts.append(f"⏱ {total:.1f}s (retrieval {retrieval_seconds:.1f}s · generating {generation_seconds:.1f}s…)")
        if tokens_out:
            parts.append(f"🔢 ~{tokens_out:,} tokens out")
    return SEPARATOR.join(parts)
