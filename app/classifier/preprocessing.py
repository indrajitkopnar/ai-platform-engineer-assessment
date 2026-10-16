"""Cleaning applied to every support message — identically at training and at prediction time.

Order matters:
1. strip HTML (emails arrive as HTML),
2. cut quoted reply chains and signatures (they repeat old content and add noise),
3. mask personal data BEFORE the text is logged or sent to any external LLM (privacy / GDPR),
4. normalise whitespace and case.
"""

from __future__ import annotations

import html
import re

_HTML_TAG = re.compile(r"<[^>]+>")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")
_LONG_NUMBER = re.compile(r"\b(?:\d[\s-]?){9,19}\b")  # phone, card and account numbers
_REPLY_HEADER = re.compile(r"^(on .+ wrote:|-----original message-----|from: .+)$", re.I)
_SIGNOFF = re.compile(r"^(regards|best regards|kind regards|thanks|thank you|cheers|br)[,!.]?$", re.I)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities (&amp; -> &). Called by: clean_message()."""
    return html.unescape(_HTML_TAG.sub(" ", text))


def remove_quoted_and_signature(text: str) -> str:
    """Keep only the newest message: stop at a reply header, a sign-off line or a '-- ' signature.
    Called by: clean_message()."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if _REPLY_HEADER.match(stripped) or _SIGNOFF.match(stripped) or stripped == "--":
            break
        kept.append(line)
    return "\n".join(kept)


def mask_pii(text: str) -> str:
    """Replace emails, IBANs and long numbers (phone/card/account) with placeholders so personal data
    is never logged or sent to an LLM. Called by: clean_message()."""
    text = _EMAIL.sub("<EMAIL>", text)
    text = _IBAN.sub("<IBAN>", text)
    return _LONG_NUMBER.sub("<NUMBER>", text)


def clean_message(subject: str, body: str) -> str:
    """Subject + body -> one cleaned, PII-masked, lower-cased string.
    Called by: Ticket.text (training) and TicketClassifier.classify() (prediction) — same code both times."""
    text = f"{subject}\n{body}" if subject else body
    text = remove_quoted_and_signature(strip_html(text))
    text = mask_pii(text)
    return re.sub(r"\s+", " ", text).strip().lower()
