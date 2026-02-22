# Wakili Chatbot API ⚖️

Arabic legal assistant powered by a hybrid RAG pipeline over Egyptian legal texts, exposed as a FastAPI service.

## Overview

This project provides:
- FastAPI backend for legal Q&A
- Hybrid retrieval (semantic + BM25 + metadata filtering)
- RRF fusion + reranking before generation
- Groq LLM integration for answer generation
- Conversation history per user/session
- Deployment options for Modal, Docker, and cloud VM setups

## Tech Stack

- FastAPI + Uvicorn
- LangChain ecosystem
- ChromaDB vector store
- Hugging Face embeddings + reranker
- Groq chat model
- Python 3.11

## Project Structure

```text
Wakili-chatbot/
├── app/
│   ├── main.py             # FastAPI app and routes
│   ├── rag_pipeline.py     # Retrieval + reranking + generation chain
│   ├── config.py           # Environment-driven settings
│   ├── schemas.py          # Request/response models
│   ├── history.py          # In-memory / Modal Dict conversation history
│   ├── deps.py             # Chain lifecycle (lazy load + reload)
│   └── utils.py
├── data/                   # Legal JSON corpus used for indexing
├── chroma_db/              # Persisted vector DB
├── reranker/               # Optional local reranker tokenizer/config files
├── deployment/
│   ├── azure/
│   ├── digitalocean/
│   ├── modal/
│   └── oracle/
├── modal_app.py            # Modal ASGI deployment entry
├── Dockerfile
├── requirements.txt
├── requirements-modal.txt
└── DEPLOY_NOW.md
```

## API Endpoints

Base app: `app.main:app`

- `GET /health` → service health
- `POST /ask` → ask legal question
- `GET /history?user_id=...&session_id=...` → get conversation history
- `POST /clear-history?user_id=...&session_id=...` → clear history
- `POST /reload` → rebuild RAG chain in-process

Interactive docs when running locally:
- `GET /docs`
- `GET /redoc`

## Request Example

`POST /ask`

```json
{
  "query": "ما الطبيعة القانونية لحق العمل؟",
  "user_id": "u_demo",
  "session_id": "s_demo_001",
  "include_sources": true,
  "eastern_arabic_numerals": false
}
```

## Environment Variables

Create a local `.env` file in the project root.

Required:
- `GROQ_API_KEY`

Common optional settings:
- `GROQ_MODEL_NAME` (default: `llama-3.3-70b-versatile`)
- `CORS_ALLOWED_ORIGINS` (comma-separated, default: `*`)
- `RERANKER_MODEL_PATH`
- `PRELOAD_CHAIN_ON_STARTUP` (`true`/`false`)
- `HISTORY_BACKEND` (`memory` or `modal_dict`)
- `HISTORY_MODAL_DICT_NAME`

Retrieval tuning options:
- `SEMANTIC_K`, `BM25_K`, `META_K`
- `HYBRID_TOP_K`, `RRF_K`, `RERANKER_TOP_N`
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TOP_P`

## Local Development

### 1) Create and activate virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Configure `.env`

Example minimal file:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CORS_ALLOWED_ORIGINS=*
PRELOAD_CHAIN_ON_STARTUP=true
```

### 4) Run API

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Quick test

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

## Deployment

### Modal (recommended in this repo)

Use the ready guide in `DEPLOY_NOW.md` and deployment scripts in `deployment/modal/`.

Main entrypoint:
- `modal_app.py`

### Docker

A `Dockerfile` is provided.

Build and run:

```powershell
docker build -t wakili-api .
docker run -p 8000:8000 --env-file .env wakili-api
```

### VM / Nginx service templates

See:
- `deployment/digitalocean/`
- `deployment/oracle/`
- `deployment/azure/`

## Data Notes

- The RAG pipeline expects JSON legal data in `data/`.
- `chroma_db/` stores persisted vector embeddings.
- On startup, chain loading may take time if models/indexes need initialization.

## Security & Git Hygiene

Sensitive/local files are excluded by `.gitignore` (for example `.env`, local DB files, keys/certs, caches, logs).

Before push, always check:

```powershell
git status
```

## Branch Upload (ahmed-abdelsalam)

```powershell
git checkout -B ahmed-abdelsalam
git add .
git commit -m "docs: rewrite README for current FastAPI project"
git push -u origin ahmed-abdelsalam
```

## License

Educational graduation project.
