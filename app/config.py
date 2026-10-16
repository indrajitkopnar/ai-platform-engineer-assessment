"""Application settings, read once from environment variables (or a local .env file).

Every tunable lives here so the same code runs on a laptop, in CI and on a server,
with only environment variables changing (12-factor style).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """All runtime settings in one immutable object. Read via the module-level `settings`
    by llm.py, classifier/service.py, chatbot/service.py and main.py."""
    # LLM gateway: "mock" runs fully offline; "openai_compatible" works with OpenAI,
    # Azure OpenAI (v1 endpoint), Ollama, vLLM or a LiteLLM proxy in front of any provider.
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

    # Scenario 1 — classifier
    classifier_model_path: Path = PROJECT_ROOT / os.getenv("CLASSIFIER_MODEL_PATH", "models/ticket_classifier.joblib")
    classifier_data_path: Path = PROJECT_ROOT / os.getenv("CLASSIFIER_DATA_PATH", "data/support_tickets.csv")
    # Below this confidence the fast model defers to the LLM; below it again → human review.
    classifier_confidence_threshold: float = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.50"))

    # Scenario 2 — chatbot
    docs_path: Path = PROJECT_ROOT / os.getenv("DOCS_PATH", "data/engineering_docs")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    # Minimum relevance of the best chunk before we even ask the LLM (anti-hallucination guardrail).
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.10"))


settings = Settings()
