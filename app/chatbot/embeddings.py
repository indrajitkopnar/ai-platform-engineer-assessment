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
    """Interface for turning text into vectors. Used by HybridRetriever; implementations below."""
    def fit(self, texts: list[str]) -> None:
        """Prepare the embedder on the corpus (TF-IDF learns its vocabulary; real models do nothing).
        Called by: HybridRetriever.__init__()."""
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised vector per text. Called by: HybridRetriever.__init__() (chunks) and
        HybridRetriever.search() (the question)."""


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so a dot product equals cosine similarity. Called by: both embedders' embed()."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class LocalTfidfEmbedder:
    """Offline embedder based on TF-IDF word weights — no download, no API key."""
    def __init__(self) -> None:
        """Create the TF-IDF vectorizer (1-2 word phrases, English stop words removed)."""
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary and word weights from the chunks. Called by: HybridRetriever.__init__()."""
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts to normalised TF-IDF vectors. Called by: HybridRetriever.__init__() and search()."""
        return _normalise(self.vectorizer.transform(texts).toarray())


class OpenAICompatibleEmbedder:
    """Real semantic embeddings from any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, LiteLLM)."""
    def __init__(self) -> None:
        """Create the HTTP client using the gateway settings and EMBEDDING_MODEL."""
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.http = httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout_seconds,
                                 headers={"Authorization": f"Bearer {settings.llm_api_key}"})

    def fit(self, texts: list[str]) -> None:  # pre-trained model: nothing to fit
        """Nothing to learn — the model is pre-trained. Kept so both embedders share one interface."""
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches of 64 via the API and normalise. Called by: HybridRetriever.__init__() and search()."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):  # batch requests
            response = self.http.post("/embeddings", json={"model": self.model, "input": texts[i:i + 64]})
            response.raise_for_status()
            vectors.extend(item["embedding"] for item in response.json()["data"])
        return _normalise(np.array(vectors, dtype=float))


def get_embedder() -> Embedder:
    """Pick the embedder from EMBEDDING_PROVIDER. Called by: ChatService.__init__()."""
    if os.getenv("EMBEDDING_PROVIDER", "local_tfidf") == "openai_compatible":
        return OpenAICompatibleEmbedder()
    return LocalTfidfEmbedder()
