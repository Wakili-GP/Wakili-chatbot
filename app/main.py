# ============================================
# file: app/main.py
# ============================================
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .deps import get_chain, reload_chain
from .history import add_to_history, clear_history, get_history
from .schemas import (
    AskRequest,
    AskResponse,
    ClearHistoryResponse,
    HistoryResponse,
    SessionRequest,
    SessionResponse,
    SourceDoc,
)
from .utils import convert_to_eastern_arabic, format_chat_history

logger = logging.getLogger(__name__)


# ─── Lightweight TTL-LRU cache for /ask responses ────────────────

class _ResponseCache:
    """Thread-safe LRU cache with TTL for RAG responses.
    Avoids re-running the full pipeline for identical (query, session) pairs.
    """
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self._maxsize = maxsize
        self._ttl = ttl                         # seconds
        self._cache: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _key(query: str, session_id: str) -> str:
        return hashlib.md5(f"{query}||{session_id}".encode()).hexdigest()

    def get(self, query: str, session_id: str) -> Optional[Dict[str, Any]]:
        k = self._key(query, session_id)
        entry = self._cache.get(k)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > self._ttl:
            self._cache.pop(k, None)
            return None
        self._cache.move_to_end(k)
        return data

    def put(self, query: str, session_id: str, data: Dict[str, Any]) -> None:
        k = self._key(query, session_id)
        self._cache[k] = (time.time(), data)
        self._cache.move_to_end(k)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


_ask_cache = _ResponseCache(
    maxsize=settings.response_cache_maxsize,
    ttl=settings.response_cache_ttl,
)

app = FastAPI(title="Legal RAG API", version="2.0.0")

# Optional: allow frontend calls
allow_any_origin = "*" in settings.cors_allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=not allow_any_origin,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    preload = os.getenv("PRELOAD_CHAIN_ON_STARTUP", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if preload:
        get_chain()


# ─── Session management ──────────────────────────────────────────

@app.post("/session", response_model=SessionResponse)
def create_session(payload: SessionRequest = SessionRequest()):
    """Frontend calls this to get a new session_id (and optionally a user_id)."""
    user_id = payload.user_id or f"user_{uuid.uuid4().hex[:12]}"
    session_id = f"sess_{uuid.uuid4().hex}"
    return SessionResponse(session_id=session_id, user_id=user_id)


# ─── Health & maintenance ────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reload")
def reload():
    reload_chain()
    _ask_cache.clear()
    return {"status": "reloaded"}


# ─── History ─────────────────────────────────────────────────────

@app.get("/history", response_model=HistoryResponse)
def history(user_id: str, session_id: str = "default"):
    messages = get_history(user_id=user_id, session_id=session_id)
    return HistoryResponse(user_id=user_id, session_id=session_id, history=messages)


@app.post("/clear-history", response_model=ClearHistoryResponse)
def clear(user_id: str, session_id: str = "default"):
    clear_history(user_id=user_id, session_id=session_id)
    return ClearHistoryResponse(user_id=user_id, session_id=session_id, cleared=True)


# ─── Ask (RAG) ───────────────────────────────────────────────────

def _dedupe_sources(docs) -> List[SourceDoc]:
    if not docs:
        return []
    seen = set()
    out: List[SourceDoc] = []
    for doc in docs:
        article_num = str(doc.metadata.get("article_number", "")).strip()
        if article_num and article_num in seen:
            continue
        if article_num:
            seen.add(article_num)
        out.append(
            SourceDoc(
                article_id=str(doc.metadata.get("article_id", "")) or None,
                article_number=article_num or None,
                law_name=str(doc.metadata.get("law_name", "")) or None,
                legal_nature=str(doc.metadata.get("legal_nature", "")) or None,
                keywords=str(doc.metadata.get("keywords", "")) or None,
                part=str(doc.metadata.get("part", "")) or None,
                chapter=str(doc.metadata.get("chapter", "")) or None,
                page_content=str(doc.page_content or ""),
            )
        )
    return out


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    t0 = time.perf_counter()
    chain = get_chain()
    active_session_id = payload.session_id or "default"

    # ── Check response cache (skip RAG for identical recent queries) ──
    cached = _ask_cache.get(payload.query, active_session_id)
    if cached is not None:
        logger.info("POST /ask cache HIT (%.3fs)", time.perf_counter() - t0)
        # Still record in history so conversation flow stays consistent
        add_to_history(
            user_id=payload.user_id,
            session_id=active_session_id,
            user_msg=payload.query,
            assistant_msg=cached["answer"],
        )
        return AskResponse(**cached, user_id=payload.user_id, session_id=active_session_id)

    # ── Build chat history from stored conversation ──
    t_hist = time.perf_counter()
    raw_history = get_history(user_id=payload.user_id, session_id=active_session_id)
    chat_history = format_chat_history(
        [{"role": m.role, "content": m.content} for m in raw_history],
        max_turns=settings.chat_history_turns,
    )
    logger.info("  history: %.3fs", time.perf_counter() - t_hist)

    # ── RAG pipeline (retrieval + reranking + LLM) ──
    t_rag = time.perf_counter()
    try:
        result = await anyio.to_thread.run_sync(
            lambda: chain.invoke({"input": payload.query, "chat_history": chat_history})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("  rag pipeline: %.3fs", time.perf_counter() - t_rag)

    answer = result.get("answer", "")
    sources_docs = result.get("context", []) if payload.include_sources else []
    sources = _dedupe_sources(sources_docs)

    if payload.eastern_arabic_numerals:
        answer = convert_to_eastern_arabic(answer)
        if payload.include_sources:
            for s in sources:
                s.page_content = convert_to_eastern_arabic(s.page_content)
                if s.article_number:
                    s.article_number = convert_to_eastern_arabic(s.article_number)

    # Build serializable raw dict (omit Document objects)
    safe_raw = {k: v for k, v in result.items() if k not in ("context",)}

    # ── Store in cache + history ──
    cache_payload = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "raw": safe_raw,
    }
    _ask_cache.put(payload.query, active_session_id, cache_payload)

    add_to_history(
        user_id=payload.user_id,
        session_id=active_session_id,
        user_msg=payload.query,
        assistant_msg=answer,
    )

    elapsed = time.perf_counter() - t0
    logger.info("POST /ask completed in %.2fs", elapsed)

    return AskResponse(
        answer=answer,
        user_id=payload.user_id,
        session_id=active_session_id,
        sources=sources,
        raw=safe_raw,
    )



