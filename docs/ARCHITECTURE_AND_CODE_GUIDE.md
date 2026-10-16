# Architecture and code guide

How the project is organised, what every file does, which technologies are used and why.

---

## 1. The idea in one paragraph

Two AI features — a **classifier** and a **RAG chatbot** — sit on **one shared core**: a single LLM gateway,
environment-based configuration, one API with validation and request tracing, one test suite, one evaluation
approach, one container and one CI pipeline. Adding a third AI feature means writing only its business logic;
everything around it already exists. That is the platform mindset at small scale.

```
                    ┌───────────────────────────────┐
                    │   app/main.py  (FastAPI)       │  validation · request id · latency logs
                    └──────────────┬────────────────┘
               ┌───────────────────┴───────────────────┐
     app/classifier/ (Scenario 1)            app/chatbot/ (Scenario 2)
               └───────────────────┬───────────────────┘
                    ┌──────────────┴────────────────┐
                    │ app/llm.py   LLM gateway       │  Mock (offline) | OpenAI-compatible (retries, timeout)
                    │ app/config.py settings (.env)  │
                    └───────────────────────────────┘
```

## 2. Folder structure — file by file

```
ai-platform-engineer-assessment/
├── app/
│   ├── config.py                 Reads all settings from environment variables / .env (provider, model,
│   │                             thresholds, paths). Same code everywhere; only the environment changes.
│   ├── llm.py                    LLM gateway. `LLMClient` interface with two implementations:
│   │                             • MockLLM — offline, deterministic; keyword classifier + extractive answers
│   │                             • OpenAICompatibleLLM — /chat/completions over HTTP, temperature 0,
│   │                               timeout, retry with backoff on 429/5xx only
│   ├── main.py                   FastAPI app: startup loads both services once; middleware adds a request id
│   │                             and logs latency; Pydantic request models validate input; 3 endpoints.
│   │
│   ├── classifier/               ── SCENARIO 1 ──
│   │   ├── preprocessing.py      strip HTML → drop quoted replies/signatures → mask PII → normalise
│   │   ├── model.py              load CSV, time-based split, TF-IDF + logistic regression pipeline,
│   │   │                         metrics (macro F1, per class, confusion matrix, threshold table), save/load
│   │   └── service.py            TicketClassifier: the cascade fast model → LLM (schema-validated) → human review
│   │
│   └── chatbot/                  ── SCENARIO 2 ──
│       ├── documents.py          read Markdown + front matter (title, access groups, revision); chunk by section
│       ├── embeddings.py         `Embedder` interface: LocalTfidfEmbedder (offline) | OpenAICompatibleEmbedder
│       ├── retriever.py          HybridRetriever: ACL filter → vector cosine + BM25 → Reciprocal Rank Fusion
│       ├── prompts.py            system prompt (the rules) + numbered-context user prompt
│       ├── guardrails.py         input check, retrieval gate, citation check, number grounding, safety notice
│       └── service.py            ChatService.ask(): the full RAG flow, returns answer + citations + notes
│
├── data/
│   ├── support_tickets.csv       90 synthetic labelled tickets (30 per class)
│   ├── engineering_docs/         5 sample documents; hr_salary_bands.md is restricted to the "hr" group
│   └── eval/chatbot_golden_set.jsonl   15 evaluation questions (expected source, status, key fact)
│
├── scripts/
│   ├── generate_sample_tickets.py  recreates the synthetic ticket CSV
│   ├── train_classifier.py         trains, prints metrics, saves model + metrics JSON to models/
│   ├── evaluate_chatbot.py         runs the golden set, prints 5 metrics
│   └── ask.py                      CLI demo for both features without starting the server
│
├── tests/                        18 pytest tests
│   ├── test_classifier.py        PII masking, signature removal, fast path, LLM down → review, invalid LLM JSON rejected
│   ├── test_chatbot.py           citations, ACL, refusal without LLM call, injection, invented number, fake citation
│   └── test_api.py               health, both endpoints, request id header, input validation (422)
│
├── docs/                         this guide, run guide, scenario write-ups, deployment guide
├── Dockerfile                    slim Python image, trains model at build time, non-root user, health check
├── docker-compose.yml            one-command local/server run
├── .github/workflows/ci.yml      tests + training + chatbot evaluation + image build on every push
├── .env.example                  every setting documented; copy to .env
├── requirements.txt              pinned dependency versions (reproducible installs)
└── pyproject.toml                project metadata + pytest configuration
```
`models/` is created by training and is not committed — models are build artefacts, rebuilt from data + code.

## 3. Request flows

**`POST /v1/classify`**
```
JSON body → Pydantic validation (422 if invalid)
→ clean_message()          HTML, quotes, signatures removed, PII masked
→ model.predict_proba()    TF-IDF + logistic regression
→ confidence ≥ 0.5 ?  ─yes→  {"category", "confidence", "source": "fast_model"}
        │no
→ llm.complete(json_mode)  → LLMLabel.model_validate_json()  (enum + range enforced)
→ valid and confident ? ─yes→ {"source": "llm"}
        │no / error / timeout
→ {"category": "needs_review", "source": "human_review"}
```

**`POST /v1/chat`**
```
JSON body → validation
→ check_input()            → "blocked" if injection / too long
→ retriever.search()       ACL filter → cosine + BM25 → RRF → top 4
→ retrieval_is_sufficient() → "refused" (LLM not called) if nothing relevant
→ llm.complete(SYSTEM_PROMPT, numbered context + question)
→ check_citations() → check_numbers_grounded()  → "refused" with a note if either fails
→ {"status": "answered", "answer", "citations": [source, section, revision, relevance], "notes"}
```

## 4. Tech stack — what and why

| Technology | Used for | Why this choice |
|---|---|---|
| **Python 3.10** | Everything | The standard language for ML and AI tooling |
| **FastAPI** + **Uvicorn** | HTTP API | Type-hint based validation, automatic OpenAPI/Swagger docs at `/docs`, fast, widely used for ML serving |
| **Pydantic v2** | Request/response models; validating LLM JSON | One schema definition gives validation, documentation and safe parsing of model output |
| **scikit-learn** | TF-IDF features, logistic regression, metrics | Proven, fast on CPU, ideal for a strong, explainable baseline |
| **NumPy** | Vector maths for cosine similarity | Standard, vectorised (no Python loops) |
| **httpx** | Calls to LLM and embedding APIs | Modern HTTP client with timeouts |
| **joblib** | Saving the trained pipeline | The standard way to persist scikit-learn models |
| **pytest** | Tests | Simple, standard; FastAPI's TestClient for API tests |
| **Docker** / **Compose** | Packaging and running | Same artefact on a laptop, a server or Kubernetes |
| **GitHub Actions** | CI | Tests, training and evaluation run on every push |
| **OpenAI-compatible API** | LLM + embeddings | One interface for OpenAI, Azure OpenAI, Ollama, vLLM and LiteLLM — no vendor lock-in |

**Deliberately not used:** LangChain/LlamaIndex (the RAG flow is ~200 lines here and every step is visible and
testable — easier to reason about and explain; a framework is a fine choice at larger scale), and a vector
database (an in-memory matrix is enough for five documents; production would use pgvector or OpenSearch).

## 5. Design principles in the code

- **Interfaces at the boundaries** — `LLMClient` and `Embedder` are `Protocol`s; swapping providers, or using fakes in
  tests, needs no change to business logic.
- **Fail safe, never crash** — LLM outage or invalid output → human review (classifier) or refusal (chatbot).
- **Configuration, not code changes** — provider, model, thresholds and paths are environment variables.
- **Same preprocessing in training and serving** — one function, used by both.
- **Guardrails in code, not only in the prompt** — prompts can be ignored or manipulated; code checks cannot.
- **Measure everything** — classifier metrics and the chatbot golden set are scripts that also run in CI.

## 6. How this becomes part of an enterprise AI platform

| This repo | Platform equivalent |
|---|---|
| `app/llm.py` | Central LLM gateway: routing, fallback, quotas per team, cost tracking, PII redaction, audit logs |
| `app/chatbot/` | "RAG as a service": teams register a document collection; the platform handles ingestion, indexing, access control and evaluation |
| `scripts/train_classifier.py` + metrics JSON | Training pipeline with experiment tracking and a model registry (e.g. MLflow); promotion only if the new model beats the current one |
| `scripts/evaluate_chatbot.py` | Shared evaluation service with golden sets per use case, run in CI |
| request-id middleware + logs | OpenTelemetry traces, metrics and dashboards per service |
| Dockerfile + CI | Standard build/deploy template every AI service uses |
