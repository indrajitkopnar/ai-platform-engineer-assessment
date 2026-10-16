"""Hybrid retrieval: semantic vector search + BM25 keyword search, fused with Reciprocal Rank Fusion.

Why hybrid: vector search understands meaning ("machine gets too hot" ≈ "high discharge
temperature") but can miss exact tokens; keyword search nails exact tokens — alarm codes (E104),
part numbers, table names — but misses paraphrases. Running both and fusing the rankings is more
robust than either alone.

Why RRF: the two scores live on different scales; RRF only uses ranks: score = Σ 1 / (60 + rank).

Access control is applied INSIDE the search (only chunks the user may see are ranked), never by
filtering the answer afterwards.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.chatbot.documents import Chunk
from app.chatbot.embeddings import Embedder

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())


@dataclass
class SearchHit:
    chunk: Chunk
    relevance: float      # semantic similarity of this chunk to the question (0..1), used by guardrails
    fused_score: float


class BM25:
    """Okapi BM25 keyword scoring (k1 = term-frequency saturation, b = length normalisation)."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = documents, k1, b
        self.avg_len = sum(map(len, documents)) / max(len(documents), 1)
        self.doc_freq = Counter(term for doc in documents for term in set(doc))
        self.tf = [Counter(doc) for doc in documents]

    def score(self, query: list[str], index: int) -> float:
        n, total = len(self.docs), 0.0
        length = len(self.docs[index])
        for term in query:
            freq = self.tf[index].get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
            total += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return total


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        self.vectors = embedder.embed(texts)                 # the "vector index" (in memory for this demo)
        self.bm25 = BM25([tokenize(t) for t in texts])

    def search(self, question: str, user_groups: set[str], top_k: int = 4) -> list[SearchHit]:
        allowed = [i for i, c in enumerate(self.chunks) if c.access & user_groups]   # ACL filter first
        if not allowed:
            return []
        similarity = self.vectors[allowed] @ self.embedder.embed([question])[0]     # cosine (normalised vectors)
        query_terms = tokenize(question)
        keyword = np.array([self.bm25.score(query_terms, i) for i in allowed])

        vector_rank = [allowed[i] for i in np.argsort(-similarity)]
        keyword_rank = [allowed[i] for i in np.argsort(-keyword) if keyword[i] > 0]
        fused: dict[int, float] = {}
        for ranking in (vector_rank, keyword_rank):
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        sim_by_index = dict(zip(allowed, similarity))
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [SearchHit(self.chunks[i], float(sim_by_index[i]), fused[i]) for i in best]
