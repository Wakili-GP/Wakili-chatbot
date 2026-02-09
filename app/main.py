# ============================================
# file: app/main.py
# ============================================
from __future__ import annotations

from typing import List

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .deps import get_chain, reload_chain
from .schemas import AskRequest, AskResponse, SourceDoc
from .utils import convert_to_eastern_arabic

app = FastAPI(title="Legal RAG API", version="1.0.0")

# Optional: allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # preload once
    get_chain()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reload")
def reload():
    reload_chain()
    return {"status": "reloaded"}


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
    chain = get_chain()

    try:
        # LangChain invoke is sync; run in worker thread
        result = await anyio.to_thread.run_sync(chain.invoke, payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

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

    return AskResponse(answer=answer, sources=sources, raw=result)



