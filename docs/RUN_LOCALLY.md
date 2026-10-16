# Run it locally

Everything works **offline** with no API key (mock LLM + local embedder). Python **3.10 or newer**.

## 1. Set up (once)

**Windows (PowerShell or cmd):**
```powershell
cd ai-platform-engineer-assessment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
cd ai-platform-engineer-assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You know the virtual environment is active when the prompt starts with `(.venv)`.

## 2. Train the classifier (Scenario 1)

```bash
python scripts/train_classifier.py
```
Prints macro F1, per-class precision/recall, the confusion matrix and the **threshold table** (how many
messages are handled automatically at each confidence level, and how accurate they are). Saves
`models/ticket_classifier.joblib`. *(The API also trains automatically on first start if the model is missing.)*

## 3. Run the tests and the chatbot evaluation

```bash
python -m pytest                     # 18 tests
python scripts/evaluate_chatbot.py   # 15 golden questions: retrieval, status, citations, key facts, refusals
```
On Windows, if the console shows odd characters, run `set PYTHONIOENCODING=utf-8` first.

## 4. Try it from the command line (no server)

```bash
python scripts/ask.py classify "I was charged twice for my last invoice"
python scripts/ask.py classify "The compressor stops and shows alarm E104"
python scripts/ask.py classify "My bill looks strange this month"         # fast model unsure → LLM fallback ("source": "llm")
python scripts/ask.py classify "Can you send me pricing for a new dryer"  # still unsure → "needs_review" (human)
python scripts/ask.py chat "What does alarm E104 mean?"
python scripts/ask.py chat "When are salary bands reviewed?"            # refused: engineering users cannot see HR docs
python scripts/ask.py chat "When are salary bands reviewed?" --groups hr # answered for HR
python scripts/ask.py chat "What is the capital of France?"             # refused: not in the documents
```

## 5. Start the API

```bash
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000/docs** — the interactive Swagger UI lets you call every endpoint from the browser.

**Example requests — bash / Git Bash / Linux:**
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/classify -H "Content-Type: application/json" \
  -d '{"subject": "Invoice", "body": "We were charged twice for invoice 4471, please refund."}'

curl -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"question": "How often must the oil be changed on the AX-200?", "user_groups": ["engineering"]}'
```

**Example requests — PowerShell:**
```powershell
Invoke-RestMethod http://localhost:8000/health

Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/classify -ContentType "application/json" `
  -Body '{"subject": "Invoice", "body": "We were charged twice for invoice 4471, please refund."}'

Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/chat -ContentType "application/json" `
  -Body '{"question": "What does alarm E212 mean?", "user_groups": ["service"]}'
```

**Example response from `/v1/chat`:**
```json
{
  "status": "answered",
  "answer": "The oil must be changed every 4000 running hours or once a year, whichever comes first. [1]",
  "citations": [{"id": 1, "source": "ax200_service_manual.md", "section": "Maintenance intervals",
                 "revision": "2026-03", "relevance": 0.24}],
  "notes": []
}
```

## 6. Use a real LLM (optional)

Copy `.env.example` to `.env` and set:

**OpenAI (or any OpenAI-compatible provider / LiteLLM proxy):**
```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-3-small
```

**A local open model with Ollama (no data leaves the machine):**
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```
```
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=nomic-embed-text
```
After switching embeddings, run `python scripts/evaluate_chatbot.py` and re-tune `MIN_RELEVANCE` — real
embeddings produce higher similarity scores than the offline TF-IDF embedder.

## 7. Run with Docker

```bash
docker compose up --build
# → http://localhost:8000/docs
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Use `py` on Windows, or install Python 3.10+ and tick "Add to PATH" |
| `ModuleNotFoundError: app` | Run commands from the project root folder |
| Port 8000 already in use | `uvicorn app.main:app --port 8001` |
| Model file missing | `python scripts/train_classifier.py` (or just start the API — it trains automatically) |
| Unicode errors in the Windows console | `set PYTHONIOENCODING=utf-8` |
