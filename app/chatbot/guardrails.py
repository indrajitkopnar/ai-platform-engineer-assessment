"""Guardrails against hallucination and misuse — layered, because no single check is enough.

Before the LLM:
  1. input checks      — length limit and obvious prompt-injection attempts are rejected;
  2. retrieval gate    — if the best chunk is not relevant enough, refuse WITHOUT calling the LLM
                         (most hallucinations start as "the model had nothing to go on").
After the LLM:
  3. citation check    — the answer must cite, and only cite sources it was actually given;
  4. number grounding  — every number in the answer (limits, intervals, codes) must appear in the
                         cited sources; an invented "6000 hours" is caught here;
  5. safety notice     — answers based on safety documents point to the official procedure.
"""

from __future__ import annotations

import re

from app.chatbot.retriever import SearchHit
from app.llm import REFUSAL

MAX_QUESTION_CHARS = 1000
_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)|system prompt|you are now|"
    r"disregard (the )?(rules|instructions)|reveal (your|the) (prompt|instructions)",
    re.I,
)


def check_input(question: str) -> str | None:
    """Return a rejection reason, or None if the question may proceed. Called by: ChatService.ask()."""
    if not question.strip():
        return "empty question"
    if len(question) > MAX_QUESTION_CHARS:
        return f"question longer than {MAX_QUESTION_CHARS} characters"
    if _INJECTION.search(question):
        return "possible prompt injection"
    return None


def retrieval_is_sufficient(hits: list[SearchHit], min_relevance: float) -> bool:
    """Retrieval gate: True only if the best chunk's similarity reaches MIN_RELEVANCE; otherwise
    ChatService.ask() refuses without calling the LLM."""
    return bool(hits) and max(h.relevance for h in hits) >= min_relevance


def cited_ids(answer: str) -> set[int]:
    """Extract the [n] citation numbers from an answer. Used by the checks below and ChatService.ask()."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def check_citations(answer: str, hits: list[SearchHit]) -> str | None:
    """Return a problem description if the answer has no citations or cites a source number that
    was not provided; None if fine. Called by: ChatService.ask()."""
    if answer.strip() == REFUSAL:
        return None
    ids = cited_ids(answer)
    if not ids:
        return "answer has no citations"
    unknown = ids - set(range(1, len(hits) + 1))
    if unknown:
        return f"answer cites sources that were not provided: {sorted(unknown)}"
    return None


def check_numbers_grounded(answer: str, hits: list[SearchHit]) -> str | None:
    """Every number in the answer must appear in at least one cited chunk; returns the missing ones.
    Called by: ChatService.ask()."""
    ids = cited_ids(answer)
    cited_text = " ".join(hits[i - 1].chunk.text for i in ids if 1 <= i <= len(hits))
    numbers = re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", "", answer))
    missing = [n for n in numbers if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", cited_text)]
    return f"numbers not found in sources: {missing}" if missing else None


def safety_notice(hits: list[SearchHit], answer: str) -> str | None:
    """Return a safety reminder if the answer cites a safety document, else None. Called by: ChatService.ask()."""
    cited = cited_ids(answer)
    if any("safety" in hits[i - 1].chunk.source for i in cited if 1 <= i <= len(hits)):
        return "Safety-related: always follow the official lock-out / tag-out procedure document."
    return None
