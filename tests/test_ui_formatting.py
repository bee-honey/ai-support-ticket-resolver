"""Response-metrics line: final snapshot and the live in-progress version."""

from __future__ import annotations

from app.models.schemas import RAGResult
from ui.formatting import format_live_stats, format_stats


def _result(**overrides) -> RAGResult:
    base = dict(answer="a", retrieval_seconds=0.6, generation_seconds=2.7, input_tokens=2017, output_tokens=306)
    base.update(overrides)
    return RAGResult(**base)


def test_final_line_matches_the_snapshot_format():
    assert format_stats(_result(), "gpt-4o-mini") == (
        "🤖 gpt-4o-mini  ·  ⏱ 3.3s (retrieval 0.6s · generation 2.7s)  ·  🔢 2,323 tokens (2,017 in · 306 out)"
    )


def test_final_line_omits_model_and_tokens_when_unknown():
    line = format_stats(_result(input_tokens=None, output_tokens=None))
    assert line == "⏱ 3.3s (retrieval 0.6s · generation 2.7s)"


def test_live_line_while_retrieving_has_no_numbers_yet():
    assert format_live_stats("gpt-4o-mini", "retrieving") == "🤖 gpt-4o-mini  ·  ⏱ retrieving evidence…"


def test_live_line_while_generating_shows_running_time_and_approximate_tokens():
    line = format_live_stats("gpt-4o-mini", "generating", retrieval_seconds=0.6, generation_seconds=1.4, tokens_out=1234)
    assert line == "🤖 gpt-4o-mini  ·  ⏱ 2.0s (retrieval 0.6s · generating 1.4s…)  ·  🔢 ~1,234 tokens out"


def test_live_line_before_the_first_token_omits_the_token_count():
    line = format_live_stats("m", "generating", retrieval_seconds=0.5, generation_seconds=0.2)
    assert "tokens" not in line and "generating 0.2s…" in line
