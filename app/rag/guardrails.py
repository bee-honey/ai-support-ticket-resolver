"""Deterministic input guardrails: pure regex, no LLM call, run on every question.

This is deliberately kept separate from the prompt-injection check in
`RAGService` (see `_is_prompt_injection`), which needs an LLM and is a judgment
call. PII redaction is neither: it's a fixed set of patterns, free, and
instant, so there is no reason to spend a model call on it and no "fails
open/closed" question to resolve -- it either matches or it doesn't.

Scope is deliberately narrower than "every kind of PII a general chatbot
might see": this app supports Mesos infrastructure tickets, where real
questions routinely contain IP addresses, ports, hex hashes, and PIDs (e.g.
"connect to 10.0.2.2:8081" is a real, legitimate question, taken from an
actual ticket seen during this project). A generic PII redactor tuned for a
consumer chatbot would mangle exactly the technical detail this app depends
on to retrieve the right ticket. So: emails, phone numbers, and credit-card-
shaped numbers are redacted (unlikely to appear in a legitimate Mesos
question, and clearly PII when they do); IP addresses are deliberately left
alone.

Known scope boundary, not yet addressed: this only redacts the INCOMING
question. Real PII already sits inside the historical ticket corpus itself
(verified directly -- the scoped CSV's comments contain real committer
emails and, in at least one case, a phone number in an email signature), and
a retrieved chunk citing that ticket can still surface it verbatim in an
answer or in the Tickets browser. Closing that gap means redacting at
ingestion/display time across the vector store and UI, which is a separate,
larger data-governance task from "add a guardrail on the request path" --
tracked here rather than silently left unmentioned.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")

# Requires punctuation/space-separated groups (xxx-xxx-xxxx, optionally with a
# leading country code) rather than any 10 consecutive digits -- that shape
# doesn't collide with the ID/port/version-number formats this domain's real
# questions use (e.g. "MESOS-1234", "port 8080", "v1.2.3").
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")

# Requires the grouped 4-4-4-4 shape (dashes or spaces) rather than any run of
# 13-16 digits -- so it doesn't collide with long numeric IDs or timestamps,
# and (being decimal-only) never matches a hex commit hash or ticket key.
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b")

_REDACTIONS = (
    (_EMAIL_RE, "[redacted-email]"),
    (_CREDIT_CARD_RE, "[redacted-card-number]"),
    (_PHONE_RE, "[redacted-phone]"),
)


def redact_pii(text: str) -> str:
    """Replace email/phone/credit-card-shaped substrings with a placeholder.

    Runs before the question is embedded, sent to any LLM, or logged anywhere
    -- the redacted text IS what the rest of the pipeline sees, not just a
    display-time mask, so PII never reaches the OpenAI API or an eval trace
    in the first place.
    """
    for pattern, placeholder in _REDACTIONS:
        text = pattern.sub(placeholder, text)
    return text
