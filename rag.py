# -*- coding: utf-8 -*-
import os
import re
import json
import logging
import warnings
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------
# Global config
# ----------------------------
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
RERANKER_DIR = os.path.join(BASE_DIR, "reranker")

_qa_chain = None
_retriever = None
_vectorstore = None   # keep a handle
_doc_store = None     # article lookup store


# ----------------------------
# Helpers
# ----------------------------
def detect_law_key(question: str) -> Optional[str]:
    q = (question or "").lower()
    if "الدستور" in q:
        return "egyptian_constitution"
    if "الإجراءات الجنائية" in q:
        return "criminal_law"
    if "تقنية المعلومات" in q or "جرائم تقنية" in q:
        return "tech_crimes"
    if "قانون العمل" in q:
        return "Egyptian_Labour_Law"
    if "الأحوال الشخصية" in q:
        return "personal_status"
    if "القانون المدني" in q:
        return "civil_law"
    return None


def extract_article_number(question: str) -> Optional[str]:
    """Extracts the number from: المادة 1 / مادة رقم 1 / نص المادة 1 ..."""
    if not question:
        return None
    m = re.search(r"(?:المادة|مادة)\s*(?:رقم\s*)?(\d+)", question)
    return m.group(1) if m else None


def is_article_request(question: str) -> bool:
    """Heuristic: user wants the article text itself (not a general Q&A)."""
    q = (question or "").strip()
    triggers = ["نص", "اعرض", "اذكر", "مضمون", "ما مضمون", "ما هو مضمون", "ما نص"]
    return (("المادة" in q) or ("مادة" in q)) and any(t in q for t in triggers)


def _load_json_folder(folder_path: str) -> List[dict]:
    all_items: List[dict] = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            root = json.load(f)

        # CASE 1: dict root
        if isinstance(root, dict):
            # accept law_key or law_id
            law_key = root.get("law_key") or root.get("law_id")
            law_name = root.get("law_name")
            data = root.get("data", [])

            if not law_key or not isinstance(data, list):
                raise ValueError(f"{filename} missing law_key/law_id or data[]")

            for item in data:
                item["_law_key"] = law_key
                item["_law_name"] = law_name
                all_items.append(item)

        # CASE 2: list root
        elif isinstance(root, list):
            inferred = os.path.splitext(filename)[0]
            law_key = inferred
            law_name = inferred

            for item in root:
                if not isinstance(item, dict):
                    continue
                item["_law_key"] = law_key
                item["_law_name"] = law_name
                all_items.append(item)
        else:
            raise ValueError(f"{filename} root must be dict or list, got {type(root)}")

    return all_items


# ----------------------------
# Chunking
# ----------------------------
def _split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking with overlap.
    - chunk_size/overlap in characters
    - keeps Arabic text safe
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)

        # try to cut at a nicer boundary
        boundary = max(text.rfind("\n", start, end), text.rfind("۔", start, end), text.rfind(".", start, end))
        if boundary != -1 and boundary > start + int(chunk_size * 0.6):
            end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def _build_documents(data: List[dict]) -> Tuple[List[Document], Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    Returns:
      - chunked documents for vector search
      - article store for exact article retrieval (law_key, article_number) -> article dict
    """
    # unique articles
    unique: Dict[str, dict] = {}
    for item in data:
        key = str(item.get("article_id") or item.get("article_number"))
        unique[key] = item

    chunk_docs: List[Document] = []
    article_store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for item in unique.values():
        law_key = item["_law_key"]
        law_name = item["_law_name"]
        article_number = str(item.get("article_number", "")).strip()

        original_text = item.get("original_text") or ""
        simplified = item.get("simplified_summary") or ""

        # store for exact retrieval (no LLM)
        article_store[(law_key, article_number)] = {
            "law_key": law_key,
            "law_name": law_name,
            "article_id": item.get("article_id"),
            "article_number": article_number,
            "original_text": original_text,
            "simplified_summary": simplified,
            "legal_nature": item.get("legal_nature", ""),
            "keywords": item.get("keywords", []),
            "part": item.get("part (Bab)") or item.get("part", ""),
            "chapter": item.get("chapter (Fasl)") or item.get("chapter", ""),
            "cross_references": item.get("cross_references", []),
        }

        # chunk the original text for better retrieval
        chunks = _split_text(original_text, chunk_size=1200, overlap=150)

        for i, ch in enumerate(chunks):
            page_content = f"""[قانون: {law_name} | مادة: {article_number} | جزء: {i+1}/{len(chunks)}]
{ch}""".strip()

            metadata = {
                "law_key": law_key,
                "law_name": law_name,
                "article_id": item.get("article_id"),
                "article_number": article_number,
                "chunk_id": i,
                "chunk_total": len(chunks),
                "legal_nature": item.get("legal_nature", ""),
                "keywords": ", ".join(item.get("keywords", [])),
                "part": item.get("part (Bab)") or item.get("part", ""),
                "chapter": item.get("chapter (Fasl)") or item.get("chapter", ""),
                "cross_references": ", ".join([str(ref) for ref in item.get("cross_references", [])]),
                # store short summary in metadata to help LLM answer
                "simplified_summary": simplified,
            }

            chunk_docs.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Loaded %d articles, created %d chunks", len(unique), len(chunk_docs))
    return chunk_docs, article_store


def _docs_to_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        num = d.metadata.get("article_number", "")
        ck = d.metadata.get("chunk_id", 0)
        parts.append(f"[المادة {num} | مقطع {ck}]\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ----------------------------
# Main initializer
# ----------------------------
def initialize_rag_pipeline():
    global _retriever, _vectorstore, _doc_store

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")
    if not os.path.exists(RERANKER_DIR):
        raise FileNotFoundError(f"Reranker folder not found: {RERANKER_DIR}")

    data = _load_json_folder(DATA_DIR)
    docs, article_store = _build_documents(data)

    _doc_store = article_store

    embeddings = HuggingFaceEmbeddings(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1",
        model_kwargs={"trust_remote_code": True},
    )

    # load or build chroma
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    else:
        _vectorstore = Chroma.from_documents(
            docs, embeddings, persist_directory=CHROMA_DIR
        )

    # Retriever created dynamically per question (law-filtered)
    def make_retriever(question: str):
        law_key = detect_law_key(question)
        search_kwargs = {"k": 20}
        if law_key:
            search_kwargs["filter"] = {"law_key": law_key}

        base = _vectorstore.as_retriever(search_kwargs=search_kwargs)

        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_DIR)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=6)

        return ContextualCompressionRetriever(
            base_retriever=base,
            base_compressor=compressor
        )

    _retriever = make_retriever

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        model_kwargs={"top_p": 0.9},
    )

    system_instructions = """
أنت مساعد قانوني مصري.
التزم بالنص الموجود في "السياق التشريعي المتاح" فقط.
- لا تضف أرقاماً/تعداداً/معلومات غير موجودة في السياق.
- إذا لم تجد إجابة في السياق قل: "لا يوجد في السياق نص كافٍ للإجابة".
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("system", "السياق التشريعي المتاح:\n{context}"),
        ("human", "سؤال المستفيد:\n{input}"),
    ])

    qa_chain = prompt | llm | StrOutputParser()
    logger.info("RAG pipeline initialized successfully.")
    return qa_chain


# ----------------------------
# Public API
# ----------------------------
def get_chain():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = initialize_rag_pipeline()
    return _qa_chain


def ask(question: str) -> Dict[str, Any]:
    """
    Returns a dict that matches your AskResponse:
    {
      "answer": str|None,
      "articles": list|None,
      "sources": [...]
    }
    """
    chain = get_chain()
    retriever_factory = _retriever

    # Always retrieve chunks for context + sources
    retriever = retriever_factory(question)
    docs = retriever.invoke(question) or []

    # Build sources from retrieved chunks (dedupe by (law_key, article_number))
    sources = []
    seen = set()
    for d in docs:
        md = d.metadata
        num = md.get("article_number")
        key = (md.get("law_key"), num)
        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "article_number": num,
            "legal_nature": md.get("legal_nature"),
            "keywords": md.get("keywords"),
            "content": d.page_content[:2000],
            "metadata": md,
        })

    # ✅ ARTICLE MODE: return exact original_text from store, not LLM
    # ✅ ARTICLE MODE: return exact original_text from JSON store (no LLM)
    target_num = extract_article_number(question)
    law_key = detect_law_key(question)

    if target_num and law_key and is_article_request(question):
        article = _doc_store.get((law_key, str(target_num)))
        if article:
            original_text = article.get("original_text", "").strip()

            return {
                "answer": original_text,   # ✅ THIS is the key change
                "articles": [{
                    "law_key": article.get("law_key"),
                    "law_name": article.get("law_name"),
                    "article_number": article.get("article_number"),
                    "original_text": original_text,
                    "simplified_summary": article.get("simplified_summary"),
                    "legal_nature": article.get("legal_nature"),
                }],
                "sources": sources
            }

        return {
            "answer": "لم أجد نص المادة المطلوبة ضمن البيانات المتاحة حالياً.",
            "articles": None,
            "sources": sources
        }


    # ✅ NORMAL Q&A MODE (LLM grounded by context chunks)
    context_text = _docs_to_context(docs)
    answer = chain.invoke({"context": context_text, "input": question})

    return {
        "answer": answer,
        "articles": None,
        "sources": sources
    }
