"""Command-line demo for both features (no API server needed).

Usage:
  python scripts/ask.py chat "What does alarm E104 mean?"
  python scripts/ask.py chat "When are salary bands reviewed?" --groups hr
  python scripts/ask.py classify "I was charged twice for my last invoice"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import get_llm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["chat", "classify"])
    parser.add_argument("text")
    parser.add_argument("--groups", default="engineering", help="comma-separated access groups for chat")
    args = parser.parse_args()

    if args.mode == "chat":
        from app.chatbot.service import ChatService
        result = ChatService(get_llm()).ask(args.text, set(args.groups.split(",")))
    else:
        from app.classifier.service import TicketClassifier
        result = TicketClassifier(get_llm()).classify("", args.text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
