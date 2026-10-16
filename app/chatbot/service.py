"""The chatbot request flow — retrieval-augmented generation with guardrails.

    question + user groups
        │ 1. input guardrail (length, prompt injection)
        ▼
    hybrid search (vector + BM25, ACL filter inside the search, RRF fusion) → top-k chunks
        │ 2. retrieval gate: best chunk relevant enough?  no → refuse, LLM never called
        ▼
    prompt = rules + numbered context + question  →  LLM via the gateway (temperature 0)
        │ 3. citation check   4. number grounding   5. safety notice
        ▼
    answer + citations (source file, section, revision) + status
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from app.chatbot import guardrails
from app.chatbot.documents import load_corpus
from app.chatbot.embeddings import Embedder, get_embedder
from app.chatbot.prompts import SYSTEM_PROMPT, build_user_prompt
from app.chatbot.retriever import HybridRetriever
from app.config import Settings, settings
from app.llm import REFUSAL, LLMClient

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """One cited source in the chat response: file, section, revision and relevance score."""
    id: int
    source: str
    section: str
    revision: str
    relevance: float


class ChatAnswer(BaseModel):
    """API response for /v1/chat: status (answered / refused / blocked), answer, citations and guardrail notes."""
    status: Literal["answered", "refused", "blocked"]
    answer: str
    citations: list[Citation] = []
    notes: list[str] = []       # which guardrail fired, safety notices — shown for transparency


class ChatService:
    """Scenario 2 entry point — the RAG flow with guardrails. Created once at API startup (main.lifespan),
    by scripts/ask.py and by scripts/evaluate_chatbot.py."""
    def __init__(self, llm: LLMClient, cfg: Settings = settings, embedder: Embedder | None = None) -> None:
        """Load and chunk the documents and build the hybrid search index.
        Called by: main.lifespan(), scripts/ask.py, scripts/evaluate_chatbot.py."""
        self.llm = llm
        self.cfg = cfg
        chunks = load_corpus(cfg.docs_path)
        self.retriever = HybridRetriever(chunks, embedder or get_embedder())
        logger.info("Indexed %d chunks from %s", len(chunks), cfg.docs_path)

    def ask(self, question: str, user_groups: set[str]) -> ChatAnswer:
        """Answer one question for a user with the given access groups: input check -> search ->
        retrieval gate -> LLM -> citation and number checks -> answer with citations.
        Called by: main.chat() (POST /v1/chat), scripts/ask.py, scripts/evaluate_chatbot.py."""
        if reason := guardrails.check_input(question):
            return ChatAnswer(status="blocked", answer="This question cannot be processed.", notes=[reason])

        hits = self.retriever.search(question, user_groups, top_k=self.cfg.retrieval_top_k)
        if not guardrails.retrieval_is_sufficient(hits, self.cfg.min_relevance):
            return ChatAnswer(status="refused", answer=REFUSAL,
                              notes=["no sufficiently relevant document found (LLM not called)"])

        draft = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits)).strip()
        if draft == REFUSAL:
            return ChatAnswer(status="refused", answer=REFUSAL, notes=["model found no answer in the context"])

        for check in (guardrails.check_citations, guardrails.check_numbers_grounded):
            if problem := check(draft, hits):
                logger.warning("Answer withheld: %s", problem)
                return ChatAnswer(status="refused", answer=REFUSAL, notes=[f"draft answer withheld: {problem}"])

        cited = sorted(guardrails.cited_ids(draft))
        citations = [Citation(id=i, source=hits[i - 1].chunk.source, section=hits[i - 1].chunk.section,
                              revision=hits[i - 1].chunk.revision, relevance=round(hits[i - 1].relevance, 3))
                     for i in cited]
        notes = [n for n in [guardrails.safety_notice(hits, draft)] if n]
        return ChatAnswer(status="answered", answer=draft, citations=citations, notes=notes)
