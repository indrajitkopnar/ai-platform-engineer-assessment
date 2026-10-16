"""LLM gateway: one small interface in front of any model provider.

Why: both features (classifier fallback and chatbot) call `LLMClient.complete()` and never a
vendor SDK directly. Switching provider or model is a configuration change, and retries,
timeouts and logging live in one place — the same idea as an enterprise LLM gateway.

Two implementations:
- `MockLLM`  — deterministic and offline, so the project runs and tests pass without an API key.
- `OpenAICompatibleLLM` — the `/chat/completions` HTTP API that OpenAI, Azure OpenAI, Ollama,
  vLLM and LiteLLM all speak.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

import httpx

from app.config import Settings, settings

logger = logging.getLogger(__name__)

REFUSAL = "I couldn't find this in the available engineering documentation."


class LLMClient(Protocol):
    """Interface every LLM provider implements. TicketClassifier._ask_llm() and ChatService.ask()
    depend only on this, so providers can be swapped (or faked in tests) without code changes."""
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the model's text reply. `json_mode` asks for a JSON object."""
        ...


class OpenAICompatibleLLM:
    """Calls any OpenAI-compatible chat completions endpoint, with timeout and retries."""

    def __init__(self, cfg: Settings, max_retries: int = 3) -> None:
        """Create one reusable HTTP client with the base URL, API key and timeout from settings.
        Called by: get_llm() when LLM_PROVIDER=openai_compatible."""
        self.cfg = cfg
        self.max_retries = max_retries
        self.http = httpx.Client(base_url=cfg.llm_base_url, timeout=cfg.llm_timeout_seconds,
                                 headers={"Authorization": f"Bearer {cfg.llm_api_key}"})

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Send one chat request and return the reply text; retries 429/5xx/timeouts with backoff.
        Called by: TicketClassifier._ask_llm() (json_mode=True) and ChatService.ask()."""
        body: dict = {
            "model": self.cfg.llm_model,
            "temperature": 0,  # factual tasks: no creativity
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self.http.post("/chat/completions", json=body)
                response.raise_for_status()
                logger.info("llm_call model=%s ms=%.0f", self.cfg.llm_model, (time.perf_counter() - start) * 1000)
                return response.json()["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in (429, 500, 502, 503)
                if attempt == self.max_retries or not retryable:
                    raise
                time.sleep(2 ** (attempt - 1))  # 1s, 2s backoff for transient errors only
        raise RuntimeError("unreachable")


class MockLLM:
    """Offline stand-in that behaves like a well-instructed model.

    - JSON classification requests → keyword heuristic, returned as the same JSON a real model returns.
    - Chatbot requests → extractive answer: the sentence from the context that best matches the
      question, cited with its [n] marker; refuses when nothing matches.
    """

    _KEYWORDS = {
        "billing": ["invoice", "charge", "charged", "refund", "payment", "paid", "bill", "vat", "credit", "price"],
        "technical": ["error", "alarm", "fault", "not working", "broken", "pressure", "leak", "noise",
                      "trip", "overheat", "temperature", "firmware", "controller", "sensor", "install"],
        "general": ["hours", "address", "catalogue", "catalog", "brochure", "visit", "contact", "distributor",
                    "job", "career", "partner", "information"],
    }

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Route to the fake classifier or the fake answerer, mirroring what a real model would do.
        Called by: TicketClassifier._ask_llm() and ChatService.ask() in offline mode."""
        if json_mode:
            return self._classify(user)
        return self._answer(user)

    def _classify(self, text: str) -> str:
        """Keyword-count classification returned as the same JSON a real LLM must return.
        Called by: MockLLM.complete() when json_mode=True."""
        lower = text.lower()
        scores = {label: sum(word in lower for word in words) for label, words in self._KEYWORDS.items()}
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        if total == 0:
            return json.dumps({"category": "general", "confidence": 0.4, "reason": "no clear signal"})
        return json.dumps({"category": best, "confidence": round(0.5 + 0.5 * scores[best] / total, 2),
                           "reason": f"mentions {best}-related terms"})

    def _answer(self, prompt: str) -> str:
        """Extractive answer: pick the context sentence sharing the most words with the question and
        cite its [n]; return the refusal sentence if nothing matches well. Called by: MockLLM.complete()."""
        question = prompt.rsplit("QUESTION:", 1)[-1].strip()
        q_terms = {w for w in re.findall(r"[a-z0-9-]+", question.lower()) if len(w) > 2} - _STOPWORDS
        best_score, best_sentence, best_id = 0, "", 0
        for block in re.finditer(r"\[(\d+)\] \(source:[^\n]*\)\n(.*?)(?=\n\[\d+\] \(source:|\nQUESTION:|\Z)", prompt, re.S):
            source_id = int(block.group(1))
            text = block.group(2).split("\n", 1)[-1]  # drop the "Title > Section" line, answer from the body
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                words = set(re.findall(r"[a-z0-9-]+", sentence.lower()))
                score = len(q_terms & words)
                if score > best_score and len(sentence.split()) > 3:
                    best_score, best_sentence, best_id = score, sentence.strip(" -*#"), source_id
        if best_score < 2:
            return REFUSAL
        return f"{best_sentence} [{best_id}]"


_STOPWORDS = {"what", "how", "the", "does", "for", "and", "are", "which", "when", "who", "with", "should",
              "can", "you", "our", "this", "that", "from", "into", "about", "there", "need", "mean", "means"}


def get_llm(cfg: Settings = settings) -> LLMClient:
    """Factory: choose the provider from LLM_PROVIDER. Called by: main.lifespan(), scripts/ask.py,
    scripts/evaluate_chatbot.py."""
    if cfg.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(cfg)
    return MockLLM()
