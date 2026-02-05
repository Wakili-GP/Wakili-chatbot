# rag.py
import os
import json
import logging
import warnings
from typing import Any, Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional deps (BM25)
try:
    from rank_bm25 import BM25Okapi
    import numpy as np
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    np = None
    _HAS_BM25 = False


# ----------------------------
# Global config
# ----------------------------
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

THIS_FILE = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(THIS_FILE)

DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
RERANKER_DIR = os.path.join(BASE_DIR, "reranker")

_qa_chain = None

# Two retrievers (general vs constitution-only routing)
_retriever_general: Optional[BaseRetriever] = None
_retriever_constitution: Optional[BaseRetriever] = None

# Composite key map to avoid article_number collisions across laws
_article_map_all: Optional[Dict[str, dict]] = None

# Canonical keys (keep stable across datasets)
CANON_CONSTITUTION = "egyptian_constitution"


# ----------------------------
# Helpers (data loading)
# ----------------------------
def _load_json_folder(folder_path: str) -> List[dict]:
    """
    Loads all json files and attaches _source_file so we can infer missing law_key/law_name.
    Supports:
      - list[dict]                      (either direct articles OR a wrapper dict that contains data/articles)
      - dict with "data": list[dict]
      - dict with "articles": list[dict]
      - single dict
    """
    all_items: List[dict] = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        def attach_source(x):
            if isinstance(x, dict):
                x["_source_file"] = filename
            return x

        def flatten_wrapper(wrapper: dict):
            """If wrapper has data/articles list, flatten it and propagate top-level law meta."""
            if not isinstance(wrapper, dict):
                return
            top_lk = wrapper.get("law_key")
            top_ln = wrapper.get("law_name")

            if "data" in wrapper and isinstance(wrapper["data"], list):
                for x in wrapper["data"]:
                    if isinstance(x, dict):
                        x.setdefault("law_key", top_lk)
                        x.setdefault("law_name", top_ln)
                    all_items.append(attach_source(x))
                return

            if "articles" in wrapper and isinstance(wrapper["articles"], list):
                for x in wrapper["articles"]:
                    if isinstance(x, dict):
                        x.setdefault("law_key", top_lk)
                        x.setdefault("law_name", top_ln)
                    all_items.append(attach_source(x))
                return

            # Not a wrapper; treat as single article dict
            all_items.append(attach_source(wrapper))

        # Case 1: list
        if isinstance(obj, list):
            for elem in obj:
                # elem might be a wrapper dict (law_key/law_name/data) OR an article dict
                if isinstance(elem, dict):
                    flatten_wrapper(elem)
                else:
                    # unexpected, but keep it
                    all_items.append(elem)
            continue

        # Case 2: dict
        if isinstance(obj, dict):
            flatten_wrapper(obj)
            continue

        logger.warning("Unsupported JSON format in: %s", file_path)

    return all_items


def _canonicalize_law_key(law_key: Optional[str]) -> Optional[str]:
    if not law_key:
        return None
    lk = str(law_key).strip()
    if not lk:
        return None
    lk_low = lk.lower()

    # Normalize common variants
    if lk_low in {"egyptian_constitution", "egyptian-constitution", "constitution", "egypt_constitution"}:
        return CANON_CONSTITUTION

    return lk_low


def _infer_law_meta(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Some datasets include law_key/law_name at the top-level only (or are missing entirely).
    We infer robustly from:
      - item["law_key"]/item["law_name"]
      - article_id prefix
      - source filename
    """
    lk = item.get("law_key")
    ln = item.get("law_name")

    # If present, canonicalize law_key
    lk_canon = _canonicalize_law_key(lk)
    if lk_canon:
        return lk_canon, (str(ln).strip() if ln else None)

    src = str(item.get("_source_file") or "").lower()
    aid = str(item.get("article_id") or "").upper()

    if "constitution" in src or aid.startswith("EG-CONST"):
        return CANON_CONSTITUTION, "الدستور المصري"

    if "labour" in src or "labor" in src or aid.startswith("EG-LABOR"):
        return "egyptian_labour_law", "قانون العمل"

    if "civil" in src or aid.startswith("EG-CIVIL"):
        return "civil_law", "القانون المدني المصري"

    if "personal status" in src or "الأحوال" in src or aid.startswith("EG-PSL"):
        return "personal_status", "قانون الأحوال الشخصية"

    if "technology crimes" in src or "tech" in src or aid.startswith("EG-TECH"):
        return "tech_crimes", "قانون مكافحة جرائم تقنية المعلومات"

    if "الإجراءات" in src or "الاجراءات" in src or aid.startswith("EG-CRIM-PROC"):
        return "criminal_law", "قانون الإجراءات الجنائية"

    return None, (str(ln).strip() if ln else None)


def _dedupe_items(items: List[dict]) -> List[dict]:
    unique: Dict[str, dict] = {}
    for item in items:
        law_key, _ = _infer_law_meta(item)
        num = str(item.get("article_number", "")).strip()
        key = str(item.get("article_id") or (f"{law_key}::{num}" if law_key and num else "") or hash(json.dumps(item, ensure_ascii=False)))
        unique[key] = item
    return list(unique.values())


def _build_documents(items: List[dict]) -> Tuple[List[Document], Dict[str, dict]]:
    """
    Returns:
      - docs: LangChain Documents
      - article_map: key = "{law_key}::{article_number}" -> raw dict (for cross-ref fetching)
    """
    article_map: Dict[str, dict] = {}
    docs: List[Document] = []

    for item in items:
        law_key, law_name = _infer_law_meta(item)
        num = str(item.get("article_number", "")).strip()

        # composite key to avoid collisions
        if law_key and num:
            article_map[f"{law_key}::{num}"] = item

        cross_refs = item.get("cross_references") or []
        cross_ref_text = ""
        if isinstance(cross_refs, list) and len(cross_refs) > 0:
            cross_ref_text = "\nالمواد ذات الصلة: " + ", ".join([f"المادة {ref}" for ref in cross_refs])

        page_content = f"""
رقم المادة: {item.get('article_number')}
النص الأصلي: {item.get('original_text')}
الشرح المبسط: {item.get('simplified_summary')}{cross_ref_text}
""".strip()

        keywords_list = item.get("keywords", [])
        keywords_str = ", ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list or "")

        metadata = {
            "article_id": item.get("article_id"),
            "article_number": num,
            "legal_nature": item.get("legal_nature", ""),
            "keywords": keywords_str,
            "part": item.get("part (Bab)", item.get("part", "")),
            "chapter": item.get("chapter (Fasl)", item.get("chapter", "")),
            "cross_references": ", ".join([str(ref) for ref in (cross_refs or [])]),
            "law_key": law_key,
            "law_name": law_name,
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Loaded %d documents", len(docs))
    return docs, article_map


def _docs_to_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        num = str(d.metadata.get("article_number", "")).strip()
        law_name = (d.metadata.get("law_name") or "").strip()
        content = (d.page_content or "").strip()
        header = f"[{law_name} - المادة {num}]" if (law_name and num) else (f"[المادة {num}]" if num else "")
        parts.append(f"{header}\n{content}".strip())
    return "\n\n---\n\n".join(parts)


def _is_constitutional_question(q: str) -> bool:
    q = (q or "").strip()
    keywords = ["الدستور", "دستوري", "دستورية", "في الدستور", "مواد الدستور", "نص المادة"]
    return any(k in q for k in keywords)



def _preferred_law_keys(question: str) -> Optional[Set[str]]:
    """
    Lightweight topic routing to reduce cross-law contamination.
    Returns a set of preferred law_keys (canonical) or None (no preference).
    """
    q = (question or "").strip().lower()

    # Constitution
    if _is_constitutional_question(question):
        return {CANON_CONSTITUTION}

    # Labour / workplace
    labour_terms = [
        "قانون العمل", "العامل", "العمال", "صاحب العمل", "منشأة", "المنشأة",
        "الأجر", "الأجور", "إضراب", "الإضراب", "فصل تعسفي", "سخرة", "جبراً", "تحرش", "تنمر", "مكان العمل"
    ]
    if any(term in q for term in labour_terms):
        return {"egyptian_labour_law"}

    # Tech crimes
    tech_terms = [
        "جرائم تقنية", "تقنية المعلومات", "اختراق", "هاكر", "حساب", "بريد إلكتروني", "ابتزاز", "انتحال", "فيسبوك",
        "واتساب", "تلغرام", "نشر", "صور", "بيانات شخصية", "خصوصية", "الشبكة", "موقع", "منصة"
    ]
    if any(term in q for term in tech_terms):
        return {"tech_crimes"}

    # Criminal procedure
    proc_terms = [
        "إجراءات جنائية", "الإجراءات الجنائية", "محضر", "بلاغ", "قسم", "شرطة", "نيابة", "تحقيق", "حبس احتياطي",
        "ضبط", "تفتيش", "قبض", "تظلم"
    ]
    if any(term in q for term in proc_terms):
        return {"criminal_law"}

    # Personal status
    ps_terms = ["أحوال شخصية", "نفقة", "حضانة", "طلاق", "خلع", "رؤية", "استضافة", "مؤخر", "قائمة المنقولات"]
    if any(term in q for term in ps_terms):
        return {"personal_status"}

    # Civil law
    civil_terms = ["القانون المدني", "عقد", "التزام", "تعويض", "مسؤولية تقصيرية", "بطلان", "إبطال", "فسخ"]
    if any(term in q for term in civil_terms):
        return {"civil_law"}

    return None



def _wants_penalty(question: str) -> bool:
    """
    Returns True if the user explicitly asks about punishment/sanctions (criminal/administrative),
    so we are allowed to include "العقوبات" articles. Otherwise, prefer substantive (non-penalty) rules.
    """
    q = (question or "").strip()
    penalty_terms = [
        "عقوبة", "العقوبات", "يُعاقب", "يعاقب", "غرامة", "حبس", "سجن", "إغلاق", "غلق",
        "جزاء", "جزاءات", "تجريم", "ما عقوبة", "ما هي عقوبة", "كم غرامة"
    ]
    return any(term in q for term in penalty_terms)

def _is_procedural_question(q: str) -> bool:
    q = (q or "").strip()
    proc = ["محضر", "بلاغ", "قسم", "شرطة", "نيابة", "دعوى", "قضية", "طلاق", "نفقة", "حضانة", "إيصال", "شيك"]
    return any(k in q for k in proc)


# ----------------------------
# Custom retrievers (BM25 / Metadata / Hybrid RRF / Cross-refs)
# ----------------------------
class BM25Retriever(BaseRetriever):
    """BM25-based keyword retriever"""
    corpus_docs: List[Document]
    k: int = 15

    bm25: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not _HAS_BM25:
            raise RuntimeError("rank_bm25/numpy not installed. Install: pip install rank-bm25 numpy")
        tokenized = [doc.page_content.split() for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][: self.k]
        return [self.corpus_docs[i] for i in top_idx if scores[i] > 0]

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class MetadataFilterRetriever(BaseRetriever):
    """Metadata + light content scoring retriever"""
    corpus_docs: List[Document]
    k: int = 15

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        q = (query or "").lower()
        q_words = [w for w in q.split() if w.strip()]
        scored = []

        for doc in self.corpus_docs:
            score = 0
            keywords = (doc.metadata.get("keywords") or "").lower()
            legal_nature = (doc.metadata.get("legal_nature") or "").lower()
            part = (doc.metadata.get("part") or "").lower()
            chapter = (doc.metadata.get("chapter") or "").lower()
            content = (doc.page_content or "").lower()
            law_name = (doc.metadata.get("law_name") or "").lower()

            if any(w in keywords for w in q_words):
                score += 3
            if any(w in legal_nature for w in q_words):
                score += 2
            if any((w in part) or (w in chapter) for w in q_words):
                score += 1
            if any(w in content for w in q_words):
                score += 1
            if any(w in law_name for w in q_words):
                score += 1

            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.k]]

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class HybridRRFRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion across:
      - semantic retriever (Chroma)
      - BM25 retriever
      - metadata retriever
    """
    semantic_retriever: BaseRetriever
    bm25_retriever: Optional[BaseRetriever] = None
    metadata_retriever: Optional[BaseRetriever] = None

    beta_semantic: float = 0.5
    beta_keyword: float = 0.3
    beta_metadata: float = 0.2

    rrf_k: int = 60
    top_k: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _doc_id(self, doc: Document) -> str:
        aid = str(doc.metadata.get("article_id") or "").strip()
        if aid:
            return aid
        lk = str(doc.metadata.get("law_key") or "").strip()
        num = str(doc.metadata.get("article_number") or "").strip()
        if lk and num:
            return f"{lk}::{num}"
        return str(hash(doc.page_content or ""))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        sem_docs = self.semantic_retriever.invoke(query) or []

        bm_docs = []
        if self.bm25_retriever is not None:
            try:
                bm_docs = self.bm25_retriever.invoke(query) or []
            except Exception:
                bm_docs = []

        md_docs = []
        if self.metadata_retriever is not None:
            md_docs = self.metadata_retriever.invoke(query) or []

        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}

        def add_ranked(docs: List[Document], beta: float):
            for rank, d in enumerate(docs, start=1):
                did = self._doc_id(d)
                rrf_scores[did] = rrf_scores.get(did, 0.0) + beta / (self.rrf_k + rank)
                if did not in doc_lookup:
                    doc_lookup[did] = d

        add_ranked(sem_docs, self.beta_semantic)
        add_ranked(bm_docs, self.beta_keyword)
        add_ranked(md_docs, self.beta_metadata)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for did, _ in sorted_ids[: self.top_k]:
            if did in doc_lookup:
                out.append(doc_lookup[did])
        return out

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class CrossReferenceRetriever(BaseRetriever):
    """Fetches cross-referenced articles automatically (within the same law)."""
    base_retriever: BaseRetriever
    article_map: Dict[str, dict]

    class Config:
        arbitrary_types_allowed = True

    def _build_doc_from_item(self, item: dict) -> Document:
        law_key, law_name = _infer_law_meta(item)

        cross_refs = item.get("cross_references") or []
        cross_ref_text = ""
        if isinstance(cross_refs, list) and len(cross_refs) > 0:
            cross_ref_text = "\nالمواد ذات الصلة: " + ", ".join([f"المادة {ref}" for ref in cross_refs])

        page_content = f"""
رقم المادة: {item.get('article_number')}
النص الأصلي: {item.get('original_text')}
الشرح المبسط: {item.get('simplified_summary')}{cross_ref_text}
""".strip()

        keywords_list = item.get("keywords", [])
        keywords_str = ", ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list or "")

        metadata = {
            "article_id": item.get("article_id"),
            "article_number": str(item.get("article_number", "")).strip(),
            "legal_nature": item.get("legal_nature", ""),
            "keywords": keywords_str,
            "cross_references": ", ".join([str(ref) for ref in (cross_refs or [])]),
            "law_key": law_key,
            "law_name": law_name,
        }
        return Document(page_content=page_content, metadata=metadata)

    def _doc_uid(self, d: Document) -> str:
        aid = str(d.metadata.get("article_id") or "").strip()
        if aid:
            return aid
        lk = str(d.metadata.get("law_key") or "").strip()
        num = str(d.metadata.get("article_number") or "").strip()
        if lk and num:
            return f"{lk}::{num}"
        return str(hash(d.page_content or ""))

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        initial_docs = self.base_retriever.invoke(query) or []

        enhanced: List[Document] = []
        seen: Set[str] = set()

        # add initial docs
        for d in initial_docs:
            uid = self._doc_uid(d)
            if uid in seen:
                continue
            seen.add(uid)
            enhanced.append(d)

        # add cross refs within same law
        for d in initial_docs:
            lk = str(d.metadata.get("law_key") or "").strip()
            if not lk:
                continue

            cross_str = (d.metadata.get("cross_references") or "").strip()
            if not cross_str:
                continue

            for ref in [x.strip() for x in cross_str.split(",")]:
                if not ref:
                    continue
                key = f"{lk}::{ref}"
                if key in self.article_map:
                    new_doc = self._build_doc_from_item(self.article_map[key])
                    uid = self._doc_uid(new_doc)
                    if uid not in seen:
                        seen.add(uid)
                        enhanced.append(new_doc)

        return enhanced

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


# ----------------------------
# Retriever builder
# ----------------------------
def _build_retriever(
    docs_subset: List[Document],
    article_map_all: Dict[str, dict],
    semantic_retriever: BaseRetriever,
) -> BaseRetriever:
    bm25_retriever = None
    if _HAS_BM25:
        try:
            bm25_retriever = BM25Retriever(corpus_docs=docs_subset, k=15)
            logger.info("BM25 enabled for subset=%d", len(docs_subset))
        except Exception as e:
            logger.warning("BM25 disabled: %s", e)

    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs_subset, k=15)

    hybrid = HybridRRFRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=0.5,
        beta_keyword=0.3,
        beta_metadata=0.2,
        rrf_k=60,
        top_k=20,
    )

    cross_ref = CrossReferenceRetriever(base_retriever=hybrid, article_map=article_map_all)

    if not os.path.exists(RERANKER_DIR):
        raise FileNotFoundError(f"Reranker folder not found: {RERANKER_DIR}")

    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_DIR)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=cross_ref)


# ----------------------------
# Main initializer
# ----------------------------
def initialize_rag_pipeline():
    global _retriever_general, _retriever_constitution, _article_map_all

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    data = _load_json_folder(DATA_DIR)
    data = _dedupe_items(data)

    docs_all, article_map_all = _build_documents(data)
    _article_map_all = article_map_all

    docs_const = [d for d in docs_all if d.metadata.get("law_key") == CANON_CONSTITUTION]
    logger.info("Constitution docs=%d (law_key=%s)", len(docs_const), CANON_CONSTITUTION)

    embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")

    # Build / load ONE Chroma DB for ALL docs
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        logger.info("Loading existing Chroma DB: %s", CHROMA_DIR)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        logger.info("Building Chroma DB (first time) into: %s", CHROMA_DIR)
        vectorstore = Chroma.from_documents(docs_all, embeddings, persist_directory=CHROMA_DIR)

    # semantic retrievers: general + constitution filter
    semantic_general = vectorstore.as_retriever(search_kwargs={"k": 15})

    # Chroma metadata filtering works only if you rebuild chroma_db after this rag.py change.
    semantic_const = vectorstore.as_retriever(search_kwargs={"k": 15, "filter": {"law_key": CANON_CONSTITUTION}})

    _retriever_general = _build_retriever(docs_all, article_map_all, semantic_general)
    _retriever_constitution = _build_retriever(docs_const, article_map_all, semantic_const)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to .env")

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        model_kwargs={"top_p": 0.9},
    )

    system_instructions = """
<role>
أنت "المساعد القانوني الذكي"، خبير متخصص في الدستور المصري والقوانين الإجرائية.
مهمتك: تقديم إجابات دقيقة بناءً على "السياق التشريعي" المرفق أولاً، أو تقديم نصائح إجرائية عامة عند الضرورة.
</role>

<decision_logic>
🔴 الحالة الأولى: إذا وجدت معلومات داخل السياق تجيب على السؤال:
- استخرج الإجابة من السياق فقط.
- ابدأ الإجابة مباشرة دون مقدمات.
- وثّق الإجابة برقم المادة.
- لا تضف معلومات خارجية.

🟡 الحالة الثانية: إذا لم تجد الإجابة في السياق وكان السؤال إجرائياً:
- ابدأ بعبارة: "بناءً على الإجراءات القانونية العامة في مصر (وليس نصاً دستورياً محدداً):"
- قدّم خطوات مرقمة واضحة.
- لا تذكر أرقام مواد قانونية.

🔵 الحالة الثالثة: إذا كان السؤال دستورياً ولم تجد الإجابة في السياق:
- قل بوضوح أنه لم يرد ذكره في المواد المسترجعة.
- لا تجب من الذاكرة.

🟢 الحالة الرابعة: تحية/شكر:
- رد مهذب ومقتضب وقل: "أنا جاهز للإجابة على استفساراتك القانونية."

⚫ الحالة الخامسة: خارج النطاق:
- اعتذر ووجّه المستخدم لمجال القانون.
</decision_logic>

<formatting_rules>
- فقرات قصيرة وبينها سطر فارغ.
- لا تكرر هذه التعليمات.
- التزم بالعربية الفصحى المبسطة.
</formatting_rules>
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("system", "السياق التشريعي المتاح:\n{context}"),
        ("human", "سؤال المستفيد:\n{input}"),
    ])

    qa_chain = (prompt | llm | StrOutputParser())

    logger.info("RAG pipeline initialized successfully.")
    return qa_chain


# ----------------------------
# Public API (matches main.py)
# ----------------------------
def get_chain():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = initialize_rag_pipeline()
    return _qa_chain


def _get_retriever_for_question(question: str) -> BaseRetriever:
    get_chain()  # ensure initialized
    if _is_constitutional_question(question):
        return _retriever_constitution or _retriever_general  # fallback
    return _retriever_general


def _docs_to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        uid = (
            md.get("article_id")
            or (str(md.get("law_key") or "") + "::" + str(md.get("article_number") or ""))
            or str(hash(d.page_content or ""))
        )
        if uid in seen:
            continue
        seen.add(uid)

        num = str(md.get("article_number", "")).strip()

        sources.append({
            "law_key": md.get("law_key"),
            "law_name": md.get("law_name"),
            "article_number": num or None,
            "legal_nature": md.get("legal_nature"),
            "keywords": md.get("keywords"),
            "content": (getattr(d, "page_content", "") or "").strip()[:2000],
            "metadata": md,
        })

    return sources


def ask(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    chain = get_chain()
    retriever = _get_retriever_for_question(question)

    docs = retriever.invoke(question) or []

    # ✅ Topic routing filter: keep only documents from the most relevant law when possible.
    preferred = _preferred_law_keys(question)
    if preferred:
        filtered = [d for d in docs if (d.metadata.get("law_key") in preferred)]
        if filtered:
            docs = filtered

    # ✅ If top doc has a law_key, prefer staying within that law to avoid mixing.
    if docs:
        top_law = docs[0].metadata.get("law_key")
        if top_law:
            same_law = [d for d in docs if d.metadata.get("law_key") == top_law]
            if same_law:
                docs = same_law

    # ✅ Penalty filter:
    # If the user did NOT ask about "عقوبة/غرامة/حبس..." then avoid pulling "العقوبات" articles
    # because they often contaminate substantive Q&A (like workplace harassment).
    if docs and not _wants_penalty(question):
        non_penalty = []
        for d in docs:
            md = d.metadata or {}
            chapter = str(md.get("chapter") or "")
            part = str(md.get("part") or "")
            legal_nature = str(md.get("legal_nature") or "")
            text = (chapter + " " + part + " " + legal_nature)
            if "عقوب" in text or "قاعدة عقابية" in text or "تجريم" in text:
                continue
            non_penalty.append(d)
        if non_penalty:
            docs = non_penalty

    context_text = _docs_to_context(docs)

    # Hard guard: constitutional questions must be answered from constitution docs only.
    if _is_constitutional_question(question):
        has_const = any((d.metadata.get("law_key") == CANON_CONSTITUTION) for d in docs)
        if not has_const:
            msg = "عذراً، لم يرد ذكر لهذا الموضوع في المواد الدستورية التي تم استرجاعها في السياق الحالي."
            return msg, _docs_to_sources(docs)

    answer = chain.invoke({"context": context_text, "input": question})

    # ✅ Post-guard 1: if we have ANY useful context, forbid procedural prefix.
    if docs:
        bad_prefix = "بناءً على الإجراءات القانونية العامة في مصر (وليس نصاً دستورياً محدداً):"
        answer = (answer or "").replace(bad_prefix, "").strip()
        answer = "\n".join([line for line in answer.splitlines() if line.strip()]).strip()

        # Also remove common hallucination like: "لا توجد مواد قانونية محددة في السياق..."
        halluc_lines = [
            "لا توجد مواد قانونية محددة في السياق المرفق",
            "لا توجد مواد قانونية محددة في السياق",
            "لا توجد معلومات داخل السياق",
            "لا توجد مواد داخل السياق",
        ]
        for hl in halluc_lines:
            answer = answer.replace(hl, "").strip()
        answer = "\n".join([line for line in answer.splitlines() if line.strip()]).strip()

    return answer, _docs_to_sources(docs)

