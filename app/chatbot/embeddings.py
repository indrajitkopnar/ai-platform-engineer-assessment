"""Turn text into vectors for semantic similarity search.

- `LocalTfidfEmbedder` (default): fitted on the corpus, fully offline, no download — good enough to
  demonstrate the pipeline on a laptop or in an interview room.
- `OpenAICompatibleEmbedder`: any `/embeddings` endpoint (OpenAI, Azure OpenAI, Ollama with
  nomic-embed-text, a LiteLLM proxy). This is what production uses: real semantic embeddings,
  ideally a multilingual model.
Rule that matters in production: documents and questions must be embedded with the SAME model;
changing the model means re-indexing everything.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...   # rows are L2-normalised vectors


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class LocalTfidfEmbedder:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.vectorizer.transform(texts).toarray())


class OpenAICompatibleEmbedder:
    def __init__(self) -> None:
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.http = httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout_seconds,
                                 headers={"Authorization": f"Bearer {settings.llm_api_key}"})

    def fit(self, texts: list[str]) -> None:  # pre-trained model: nothing to fit
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):  # batch requests
            response = self.http.post("/embeddings", json={"model": self.model, "input": texts[i:i + 64]})
            response.raise_for_status()
            vectors.extend(item["embedding"] for item in response.json()["data"])
        return _normalise(np.array(vectors, dtype=float))


def get_embedder() -> Embedder:
    if os.getenv("EMBEDDING_PROVIDER", "local_tfidf") == "openai_compatible":
        return OpenAICompatibleEmbedder()
    return LocalTfidfEmbedder()
