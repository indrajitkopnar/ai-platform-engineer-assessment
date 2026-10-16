# AI Platform Engineer — Candidate Evaluation

Two AI features built on **one shared platform core** (LLM gateway, configuration, API, request
tracing, evaluation, CI and container):

| # | Scenario chosen | What it does | Details |
|---|---|---|---|
| 1 | **Support message classifier** | Routes customer messages to *billing / technical / general*, with a confidence-based cascade: fast ML model → LLM fallback → human review | [docs/SCENARIO_1_SUPPORT_TICKET_CLASSIFIER.md](docs/SCENARIO_1_SUPPORT_TICKET_CLASSIFIER.md) |
| 2 | **Engineering knowledge chatbot** | Answers engineers' questions from engineering documents (RAG) with citations, access control and layered anti-hallucination guardrails | [docs/SCENARIO_2_ENGINEERING_CHATBOT.md](docs/SCENARIO_2_ENGINEERING_CHATBOT.md) |
| 3 | *(not chosen — design notes only)* | Detecting incorrect information in documents | [docs/SCENARIO_3_DOCUMENT_VERIFICATION_DESIGN.md](docs/SCENARIO_3_DOCUMENT_VERIFICATION_DESIGN.md) |

**Why these two:** both are common enterprise patterns (routing and knowledge access), and together they
show the full lifecycle — data, preprocessing, model choice, serving, guardrails, evaluation and deployment.

**Runs fully offline by default.** A deterministic mock LLM and a local embedder mean the project runs and
all tests pass without any API key. Setting three environment variables switches to a real model
(OpenAI, Azure OpenAI, a local Ollama model, or a LiteLLM proxy).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/train_classifier.py      # trains + prints evaluation metrics
python -m pytest                        # 18 tests
python scripts/evaluate_chatbot.py      # chatbot golden-set evaluation

uvicorn app.main:app --reload           # API on http://localhost:8000  →  open http://localhost:8000/docs
```

Full step-by-step guide with example requests: **[docs/RUN_LOCALLY.md](docs/RUN_LOCALLY.md)**

---

## Architecture

```
                       ┌──────────────────────── FastAPI service (app/main.py) ────────────────────────┐
  client / helpdesk ──►│  request-id + latency logging middleware · Pydantic request validation          │
                       │                                                                                 │
  POST /v1/classify ──►│  SCENARIO 1  TicketClassifier                                                   │
                       │   clean + mask PII → TF-IDF + logistic regression ──confident──► label          │
                       │                         │ low confidence                                         │
                       │                         ▼                                                        │
                       │                     LLM (JSON, validated enum) ──confident──► label              │
                       │                         │ still unsure / invalid / provider down                 │
                       │                         ▼                                                        │
                       │                     "needs_review" → human queue                                 │
                       │                                                                                 │
  POST /v1/chat ──────►│  SCENARIO 2  ChatService (RAG)                                                  │
                       │   input guardrail → hybrid search (vector + BM25, ACL filter, RRF) → top-k chunks│
                       │   → retrieval gate → prompt with numbered sources → LLM                          │
                       │   → citation check → number-grounding check → safety notice → answer + sources  │
                       │                                                                                 │
                       │  SHARED PLATFORM CORE: app/llm.py (gateway: mock | OpenAI-compatible, retries,   │
                       │  timeouts) · app/config.py (env-based settings) · logging · tests · CI · Docker  │
                       └─────────────────────────────────────────────────────────────────────────────────┘
```

Folder-by-folder and file-by-file explanation, tech stack and design decisions:
**[docs/ARCHITECTURE_AND_CODE_GUIDE.md](docs/ARCHITECTURE_AND_CODE_GUIDE.md)**

---

## Results (offline mode, synthetic data)

**Classifier** — 90 synthetic tickets, trained on the oldest 75 %, tested on the newest 25 % (time-based split):

| Metric | Value |
|---|---|
| Macro F1 (fast model alone, every message) | **0.83** |
| At confidence threshold 0.5: share handled automatically | **83 %** |
| Accuracy on that automatically handled share | **95 %** |
| Remaining messages | go to the LLM fallback, then to human review |

**Chatbot** — 15-question golden set (11 answerable, 3 must be refused, 1 prompt injection):

| Metric | Value |
|---|---|
| Retrieval hit rate (right document in top 4) | 11/11 |
| Correct status (answer / refuse / block) | 15/15 |
| Citation accuracy | 11/11 |
| Answer contains the expected key fact | 11/11 |
| Refusals when the answer is unavailable or access is denied | 4/4 |

> These numbers come from a small synthetic dataset and are meant to show the **method**, not to claim
> production accuracy. With real data, the same scripts produce the real numbers.

---

## Repository map

```
app/
  config.py            settings from environment variables
  llm.py               LLM gateway: MockLLM (offline) + OpenAICompatibleLLM (retries, timeouts)
  main.py              FastAPI app: /health, /v1/classify, /v1/chat, request tracing
  classifier/          Scenario 1 — preprocessing.py, model.py (train/evaluate), service.py (cascade)
  chatbot/             Scenario 2 — documents.py (chunking), embeddings.py, retriever.py (hybrid + ACL),
                       prompts.py, guardrails.py, service.py (RAG flow)
data/
  support_tickets.csv          synthetic labelled tickets (generated by scripts/generate_sample_tickets.py)
  engineering_docs/*.md        sample engineering documents with access groups in a front-matter header
  eval/chatbot_golden_set.jsonl  evaluation questions with expected source, status and key fact
scripts/               train_classifier.py · evaluate_chatbot.py · ask.py (CLI demo) · generate_sample_tickets.py
tests/                 18 tests: preprocessing, cascade fallbacks, guardrails, access control, API
docs/                  scenario write-ups, architecture guide, run guide, deployment guide
Dockerfile · docker-compose.yml · .github/workflows/ci.yml · .env.example
```

## Deployment
Docker, a Linux server behind nginx with TLS, and a Kubernetes outline: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**

## Tech stack
Python 3.10 · FastAPI + Uvicorn · Pydantic v2 · scikit-learn (TF-IDF, logistic regression) · NumPy ·
httpx · pytest · Docker · GitHub Actions. LLM and embeddings through any OpenAI-compatible API.
