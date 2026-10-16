"""Evaluate the chatbot on a golden set of questions — retrieval and answer behaviour separately.

Metrics
  retrieval_hit_rate  for answerable questions: is the expected document among the top-k chunks?
  status_accuracy     did the bot answer / refuse / block exactly when it should?
  citation_accuracy   for answered questions: does the answer cite the expected document?
  answer_correctness  for answered questions: does the answer contain the expected key fact?
  refusal_precision   for questions with no answer (or forbidden by access rights): did it refuse?

Run this in CI on every change to prompts, chunking, embedding model or LLM; block the release if
scores drop. Usage:  python scripts/evaluate_chatbot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chatbot.service import ChatService  # noqa: E402
from app.llm import get_llm  # noqa: E402


def main() -> None:
    cases = [json.loads(line) for line in (ROOT / "data/eval/chatbot_golden_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    service = ChatService(get_llm())

    retrieval_hits = answerable = status_ok = citation_ok = correct = answered_expected = refusals_ok = unanswerable = 0
    for case in cases:
        groups = set(case["user_groups"])
        result = service.ask(case["question"], groups)
        status_ok += result.status == case["expected_status"]
        mark = "OK  " if result.status == case["expected_status"] else "FAIL"

        if case["expected_source"]:
            answerable += 1
            sources = {h.chunk.source for h in service.retriever.search(case["question"], groups)}
            retrieval_hits += case["expected_source"] in sources
            if case["expected_status"] == "answered":
                answered_expected += 1
                citation_ok += any(c.source == case["expected_source"] for c in result.citations)
                correct += bool(case.get("answer_contains")) and case["answer_contains"].lower() in result.answer.lower()
        else:
            unanswerable += 1
            refusals_ok += result.status in ("refused", "blocked")
        print(f"[{mark}] {result.status:<8} {case['question'][:60]:<60} | {result.answer[:70]}")

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} = {a / b:.0%}" if b else "n/a"

    print("\nretrieval_hit_rate :", pct(retrieval_hits, answerable))
    print("status_accuracy    :", pct(status_ok, len(cases)))
    print("citation_accuracy  :", pct(citation_ok, answered_expected))
    print("answer_correctness :", pct(correct, answered_expected))
    print("refusal_precision  :", pct(refusals_ok, unanswerable))


if __name__ == "__main__":
    main()
