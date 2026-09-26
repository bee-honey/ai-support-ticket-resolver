"""redact_pii: deterministic, no LLM involved -- pure regex in, string out."""

from __future__ import annotations

from app.rag.guardrails import redact_pii


def test_redacts_an_email_address():
    result = redact_pii("please reach me at jane.doe@example.com if you have questions")
    assert "jane.doe@example.com" not in result
    assert "[redacted-email]" in result


def test_redacts_a_punctuated_phone_number():
    result = redact_pii("call me at 415-555-0142 about this ticket")
    assert "415-555-0142" not in result
    assert "[redacted-phone]" in result


def test_redacts_a_grouped_credit_card_number():
    result = redact_pii("card is 4111-1111-1111-1111, charge failed")
    assert "4111-1111-1111-1111" not in result
    assert "[redacted-card-number]" in result


def test_redacts_multiple_categories_in_the_same_string():
    result = redact_pii("email jane@example.com or call 415-555-0142")
    assert "[redacted-email]" in result and "[redacted-phone]" in result


def test_does_not_redact_an_ip_address():
    # legitimate technical content in this domain -- a real Mesos ticket question,
    # e.g. "agent can't connect to 10.0.2.2:8081", must survive unredacted or
    # retrieval quality on real questions like it would silently degrade.
    text = "agent can't connect to 10.0.2.2:8081, connection refused"
    assert redact_pii(text) == text


def test_does_not_redact_a_ticket_id_or_version_number():
    text = "MESOS-1873 regressed again on v1.2.3, port 8080 unreachable"
    assert redact_pii(text) == text


def test_does_not_redact_a_hex_commit_hash():
    text = "bisected to commit 4366d551a9f3c2b0 breaking the scheduler"
    assert redact_pii(text) == text


def test_empty_string_is_a_no_op():
    assert redact_pii("") == ""


def test_text_with_no_pii_is_unchanged():
    text = "docker daemon fails to start after upgrading to mesos 1.9"
    assert redact_pii(text) == text
