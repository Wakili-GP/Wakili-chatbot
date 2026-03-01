# Wakili — Egyptian Legal AI Chatbot API ⚖️

> Arabic-first legal assistant powered by a hybrid RAG pipeline over six Egyptian legal codes, served as a FastAPI service on [Modal](https://modal.com).

---

## Architecture

```
Client (React / Postman)
    │
    ▼
FastAPI   ──►  /session   →  session_id + user_id
    │
    ▼
POST /ask ──►  Hybrid Retriever (Semantic ∥ BM25 ∥ Metadata)
                      │
                      ▼
               RRF Fusion  (Reciprocal Rank Fusion)
                      │
                      ▼
               CrossEncoder Reranker  (BAAI/bge-reranker-v2-m3)
                      │
                      ▼
               Groq LLM  (llama-3.3-70b-versatile)
                      │
                      ▼
               Formatted answer + source articles
```

### Legal Corpus

| Code | File |
|------|------|
| Egyptian Constitution | `Egyptian_Constitution_legalnature_only.json` |
| Civil Code | `Egyptian_Civil.json` |
| Labour Law | `Egyptian_Labour_Law.json` |
| Personal Status Laws | `Egyptian_Personal Status Laws.json` |
| Technology Crimes Law | `Technology Crimes Law.json` |
| Criminal Procedures | `قانون_الإجراءات_الجنائية.json` |

**Total articles indexed:** ~2 376

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/session` | Create a new session (returns `session_id` + `user_id`) |
| `POST` | `/ask` | Ask a legal question (RAG pipeline) |
| `GET` | `/history` | Get conversation history (`?user_id=...&session_id=...`) |
| `POST` | `/clear-history` | Clear conversation history |
| `POST` | `/reload` | Rebuild RAG chain + clear response cache |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### `POST /session`

```json
// Request (all fields optional)
{ "user_id": "optional-existing-id" }

// Response
{ "session_id": "sess_...", "user_id": "user_..." }
```

### `POST /ask`

```json
// Request
{
  "query": "ما هي حقوق العامل في قانون العمل؟",
  "user_id": "user_abc",
  "session_id": "sess_xyz",
  "include_sources": true,
  "eastern_arabic_numerals": false
}

// Response
{
  "answer": "...",
  "user_id": "user_abc",
  "session_id": "sess_xyz",
  "sources": [
    {
      "article_id": "EG-LABOUR-ART-1",
      "article_number": "1",
      "law_name": "قانون العمل المصري",
      "page_content": "..."
    }
  ],
  "raw": {}
}
```

---

## Performance Tuning

The `/ask` endpoint goes through: **3 parallel retrievers → RRF fusion → CrossEncoder reranking → LLM generation**.

### Current defaults (tunable via env vars)

| Parameter | Default | Env var | Effect |
|-----------|---------|---------|--------|
| Semantic K | 10 | `SEMANTIC_K` | Docs from vector search |
| BM25 K | 10 | `BM25_K` | Docs from keyword search |
| Metadata K | 8 | `META_K` | Docs from metadata index |
| Hybrid top-K | 10 | `HYBRID_TOP_K` | Docs sent to reranker |
| Reranker top-N | 5 | `RERANKER_TOP_N` | Docs sent to LLM |
| Max tokens | 768 | `LLM_MAX_TOKENS` | LLM output cap |
| LLM timeout | 60 s | `LLM_TIMEOUT` | Fail-fast on slow responses |
| Response cache TTL | 300 s | `RESPONSE_CACHE_TTL` | Skip RAG for repeated queries |
| Response cache size | 128 | `RESPONSE_CACHE_MAXSIZE` | Max cached responses |

### What speeds things up

- **Response cache** — identical `(query, session_id)` pairs hit cache (<5 ms) for 5 minutes
- **Global thread pool** — 3 retrievers run in parallel without per-request pool overhead
- **Document truncation** — long articles are trimmed to 1 200 chars before entering LLM context
- **Lower K values** — fewer candidates through the expensive reranker
- **Lower max_tokens / LLM timeout** — LLM generates and fails faster

---

## Project Structure

```
Wakili-chatbot/
├── app/
│   ├── main.py             # FastAPI app, routes, response cache
│   ├── rag_pipeline.py     # Full RAG chain builder (retrieval → reranking → LLM)
│   ├── config.py           # All settings (env-overridable dataclass)
│   ├── schemas.py          # Pydantic request/response models
│   ├── history.py          # Conversation history (memory / Modal Dict)
│   ├── deps.py             # Thread-safe chain singleton
│   └── utils.py            # Arabic tokenizer, numeral converter, chat history formatter
├── data/                   # Legal JSON corpus (6 files)
├── chroma_db/              # Pre-built vector DB (~2 376 vectors)
├── reranker/               # Reranker tokenizer files (weights from HF cache)
├── deployment/
│   └── modal/
│       └── deploy.ps1      # Modal deploy helper script
├── archive/                # Old / unused files (not deployed)
├── modal_app.py            # Modal serverless deployment entry
├── requirements.txt        # Full local development dependencies
├── requirements-modal.txt  # Minimal production dependencies (Modal)
└── .gitignore
```

---

## Environment Variables

Create `.env` in the project root:

```env
# Required
GROQ_API_KEY=gsk_...

# Optional — defaults shown
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CORS_ALLOWED_ORIGINS=*
PRELOAD_CHAIN_ON_STARTUP=true
EMBEDDING_MODEL=Omartificial-Intelligence-Space/GATE-AraBert-v1
CHROMA_DIR=                     # auto-detected if empty
RERANKER_MODEL_PATH=            # local tokenizer dir; weights from HF cache

# History backend
HISTORY_BACKEND=memory          # or "modal_dict" on Modal
HISTORY_MODAL_DICT_NAME=wakili-history

# Retrieval tuning
SEMANTIC_K=10
BM25_K=10
META_K=8
HYBRID_TOP_K=10
RRF_K=60
RERANKER_TOP_N=5
BETA_SEMANTIC=0.60
BETA_BM25=0.20
BETA_METADATA=0.20

# LLM tuning
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=768
LLM_TOP_P=0.80
LLM_MAX_RETRIES=2
LLM_TIMEOUT=60
CHAT_HISTORY_TURNS=3

# Cache
RESPONSE_CACHE_MAXSIZE=128
RESPONSE_CACHE_TTL=300
```

---

## Local Development

```powershell
# 1. Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Add .env (see above)

# 4. Run
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Test
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

---

## Deploy to Modal

```powershell
# Install and auth
pip install modal
modal token new

# Create secret (one-time)
$Key = (Get-Content .env | Select-String "^GROQ_API_KEY").ToString().Split("=",2)[1].Trim().Trim('"')
modal secret create wakili-secrets GROQ_API_KEY="$Key" GROQ_MODEL_NAME="llama-3.3-70b-versatile"

# Deploy
modal deploy modal_app.py

# Test
$Base = "https://ahmd-mohmd--wakili-api-fastapi-app.modal.run"
Invoke-RestMethod "$Base/health"
```

Container stays warm for **30 minutes** after the last request (`scaledown_window=1800`).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.115+ |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | GATE-AraBert-v1 (HuggingFace) |
| Reranker | bge-reranker-v2-m3 (CrossEncoder) |
| LLM | Groq — Llama 3.3 70B |
| Keyword Search | BM25 (rank-bm25) |
| Deployment | Modal (serverless) |
| Runtime | Python 3.11 |

---

## License

Educational graduation project — Faculty of Engineering.
