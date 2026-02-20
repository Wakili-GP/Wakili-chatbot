# ============================================
# file: app/rag_pipeline.py
# ============================================
from __future__ import annotations

import json
import logging
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set

import numpy as np
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from rank_bm25 import BM25Okapi

from .config import Settings
from .utils import arabic_tokenize

os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_json_folder(folder_path: str) -> List[dict]:
    all_items: List[dict] = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".json"):
            continue
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        wrapper_law_name = ""

        if isinstance(obj, list):
            articles: List[dict] = []
            for entry in obj:
                if isinstance(entry, dict) and "data" in entry and isinstance(entry["data"], list):
                    wrapper_law_name = entry.get("law_name", "")
                    for art in entry["data"]:
                        art.setdefault("_law_name", wrapper_law_name)
                    articles.extend(entry["data"])
                elif isinstance(entry, dict) and "articles" in entry and isinstance(entry["articles"], list):
                    wrapper_law_name = entry.get("law_name", "")
                    for art in entry["articles"]:
                        art.setdefault("_law_name", wrapper_law_name)
                    articles.extend(entry["articles"])
                elif isinstance(entry, dict):
                    if not entry.get("_law_name"):
                        aid = entry.get("article_id", "")
                        entry["_law_name"] = "الدستور المصري" if "CONST" in str(aid).upper() else ""
                    articles.append(entry)
            all_items.extend(articles)
        elif isinstance(obj, dict):
            wrapper_law_name = obj.get("law_name", "")
            if "data" in obj and isinstance(obj["data"], list):
                for art in obj["data"]:
                    art.setdefault("_law_name", wrapper_law_name)
                all_items.extend(obj["data"])
            elif "articles" in obj and isinstance(obj["articles"], list):
                for art in obj["articles"]:
                    art.setdefault("_law_name", wrapper_law_name)
                all_items.extend(obj["articles"])
            else:
                obj.setdefault("_law_name", wrapper_law_name)
                all_items.append(obj)
        else:
            logger.warning("Unsupported JSON format in: %s", file_path)

    return all_items


def build_qa_chain(settings: Settings, conversation_history: List[dict] = None):
    """
    Builds and returns:
      qa_chain: Runnable that returns {"context": [Document...], "input": str, "answer": str}
    
    Args:
        settings: Configuration settings
        conversation_history: Optional list of previous messages for context
    """
    if not os.path.exists(settings.data_dir):
        raise FileNotFoundError(f"Data folder not found: {settings.data_dir}")

    data = _load_json_folder(settings.data_dir)

    # de-dup
    unique: Dict[str, dict] = {}
    for item in data:
        key = str(item.get("article_id") or item.get("article_number") or hash(json.dumps(item, ensure_ascii=False)))
        unique[key] = item
    data = list(unique.values())

    docs: List[Document] = []
    for item in data:
        article_number = item.get("article_number")
        original_text = item.get("original_text")
        simplified_summary = item.get("simplified_summary")
        if not article_number or not original_text or not simplified_summary:
            continue

        law_name = item.get("law_name") or item.get("_law_name", "")
        part_bab = item.get("part (Bab)", "")
        chapter_fasl = item.get("chapter (Fasl)", "")
        section = item.get("section", "")

        page_content = f"""القانون: {law_name}
رقم المادة: {article_number}
الباب: {part_bab}
الفصل: {chapter_fasl}
القسم: {section}
النص الأصلي: {original_text}
الشرح المبسط: {simplified_summary}"""

        metadata = {
            "article_id": item.get("article_id") or str(article_number),
            "article_number": str(article_number),
            "law_name": law_name,
            "legal_nature": item.get("legal_nature", ""),
            "keywords": ", ".join(item.get("keywords", []) or []),
            "part": part_bab,
            "chapter": chapter_fasl,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))

    if not docs:
        raise RuntimeError("No valid documents found in data folder (missing required fields).")

    embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")

    # vector store reuse
    db_exists = os.path.exists(settings.chroma_dir) and os.listdir(settings.chroma_dir)
    if db_exists:
        vectorstore = Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings,
        )
        stored_count = vectorstore._collection.count()
        if stored_count == 0 or abs(stored_count - len(docs)) > 5:
            import shutil

            shutil.rmtree(settings.chroma_dir, ignore_errors=True)
            db_exists = False

    if not db_exists:
        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=settings.chroma_dir,
        )

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.semantic_k})

    # -----------------------------
    # BM25 retriever
    # -----------------------------
    class BM25Retriever(BaseRetriever):
        corpus_docs: List[Document]
        bm25: BM25Okapi = None
        tokenized_corpus: list = None
        k: int = 10

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, **data):
            super().__init__(**data)
            self.tokenized_corpus = [arabic_tokenize(doc.page_content) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            tokenized_query = arabic_tokenize(query)
            if not tokenized_query:
                return []
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][: self.k]
            return [self.corpus_docs[i] for i in top_indices if scores[i] > 0]

        async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    bm25_retriever = BM25Retriever(corpus_docs=docs, k=settings.bm25_k)

    # -----------------------------
    # Metadata filter retriever
    # -----------------------------
    class MetadataFilterRetriever(BaseRetriever):
        corpus_docs: List[Document]
        keyword_index: Dict[str, Set[int]] = None
        law_name_index: Dict[str, Set[int]] = None
        k: int = 10

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, **data):
            super().__init__(**data)
            self.keyword_index = defaultdict(set)
            self.law_name_index = defaultdict(set)
            for idx, doc in enumerate(self.corpus_docs):
                kw_text = (
                    str(doc.metadata.get("keywords", ""))
                    + " "
                    + str(doc.metadata.get("legal_nature", ""))
                    + " "
                    + str(doc.metadata.get("part", ""))
                    + " "
                    + str(doc.metadata.get("chapter", ""))
                )
                for token in arabic_tokenize(kw_text):
                    self.keyword_index[token].add(idx)

                for token in arabic_tokenize(str(doc.metadata.get("law_name", ""))):
                    self.law_name_index[token].add(idx)

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            query_tokens = arabic_tokenize(query)
            if not query_tokens:
                return []

            scores = defaultdict(float)
            for token in query_tokens:
                for idx in self.keyword_index.get(token, set()):
                    scores[idx] += 3.0
                for idx in self.law_name_index.get(token, set()):
                    scores[idx] += 4.0

            if not scores:
                return []

            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.k]
            return [self.corpus_docs[idx] for idx, _ in top]

        async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=settings.meta_k)

    # -----------------------------
    # Hybrid RRF retriever (parallel)
    # -----------------------------
    class HybridRRFRetriever(BaseRetriever):
        semantic_retriever: BaseRetriever
        bm25_retriever: BM25Retriever
        metadata_retriever: MetadataFilterRetriever
        beta_semantic: float = 0.5
        beta_keyword: float = 0.3
        beta_metadata: float = 0.2
        k: int = 60
        top_k: int = 12

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_sem = pool.submit(self.semantic_retriever.invoke, query)
                fut_bm = pool.submit(self.bm25_retriever.invoke, query)
                fut_meta = pool.submit(self.metadata_retriever.invoke, query)

                semantic_docs = fut_sem.result()
                bm25_docs = fut_bm.result()
                metadata_docs = fut_meta.result()

            rrf_scores: Dict[str, float] = {}
            all_docs: Dict[str, Document] = {}

            for weight, doc_list in [
                (self.beta_semantic, semantic_docs),
                (self.beta_keyword, bm25_docs),
                (self.beta_metadata, metadata_docs),
            ]:
                for rank, doc in enumerate(doc_list, start=1):
                    doc_id = doc.metadata.get("article_id") or doc.metadata.get("article_number") or str(hash(doc.page_content))
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (self.k + rank)
                    all_docs.setdefault(doc_id, doc)

            sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return [all_docs[did] for did, _ in sorted_ids[: self.top_k] if did in all_docs]

        async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=0.5,
        beta_keyword=0.3,
        beta_metadata=0.2,
        k=settings.rrf_k,
        top_k=settings.hybrid_top_k,
    )

    # -----------------------------
    # Reranker (CrossEncoder)
    # -----------------------------
    if not settings.reranker_model_path:
        raise RuntimeError("RERANKER_MODEL_PATH is not set in .env (must point to your local reranker folder).")
    if not os.path.exists(settings.reranker_model_path):
        raise FileNotFoundError(f"Reranker path not found: {settings.reranker_model_path}")

    cross_encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model_path)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )

    # -----------------------------
    # LLM + prompt
    # -----------------------------
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing in .env")

    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        model_kwargs={"top_p": settings.top_p},
    )

    system_instructions = """
<role>
أنت "المساعد القانوني الذكي"، مستشار قانوني متخصص في القوانين المصرية التالية:
- الدستور المصري
- القانون المدني المصري
- قانون العمل المصري
- قانون الأحوال الشخصية المصري
- قانون مكافحة جرائم تقنية المعلومات
- قانون الإجراءات الجنائية المصري

مهمتك الأساسية: الإجابة بدقة استناداً إلى "السياق التشريعي" المرفق أدناه.
عند وجود نص قانوني في السياق، هو مصدرك الأول والأهم.

استخدم سجل المحادثة السابقة (إن وجد) لفهم السياق والإجابة بتسلسل منطقي.
</role>

<decision_logic>
حلّل سؤال المستخدم ثم اتبع أول حالة ينطبق شرطها:

━━━ الحالة ١ — الإجابة موجودة في السياق (الأولوية القصوى) ━━━
الشرط: توجد مادة أو أكثر في السياق تتناول موضوع السؤال بشكل مباشر أو وثيق الصلة.
الفعل:
• أجب من السياق مباشرةً دون مقدمات.
• وثّق كل معلومة بذكر اسم القانون ورقم المادة (مثال: «وفقاً للمادة (٥٢) من قانون العمل...»).
• استخرج ما يجيب السؤال تحديداً — لا تنسخ المادة كاملة.
• لا تضف معلومات من خارج السياق في هذه الحالة.

━━━ الحالة ٢ — السياق يغطي الموضوع جزئياً ━━━
الشرط: توجد مواد ذات صلة لكنها لا تجيب السؤال بالكامل.
الفعل:
• اذكر أولاً ما تنص عليه المواد المتاحة (مع التوثيق).
• ثم أضف توضيحاً عملياً مختصراً يساعد المستخدم، مع التنبيه بعبارة:
  «ملاحظة عملية:» أو «من الناحية التطبيقية:» قبل أي إضافة.
• لا تخترع أرقام مواد أو تنسب نصوصاً لقوانين لم ترد في السياق.

━━━ الحالة ٣ — السياق لا يحتوي الإجابة + السؤال إجرائي/عملي ━━━
الشرط: لا توجد مادة في السياق تتعلق بالموضوع، لكن السؤال عن إجراءات عملية (بلاغ، محضر، حادث، طلاق، تعامل مع الشرطة...).
الفعل:
• ابدأ بعبارة: «بناءً على الإجراءات القانونية المتعارف عليها في مصر (وليس استناداً لنص قانوني محدد من قاعدة البيانات):»
• قدم خطوات عملية مرقمة ومختصرة.
• لا تذكر أرقام مواد — لا تختـرع مراجع.
• أنهِ بـ«يُنصح بمراجعة محامٍ متخصص لتأكيد الإجراءات.»

━━━ الحالة ٤ — السياق لا يحتوي الإجابة + السؤال عن نص قانوني بعينه ━━━
الشرط: المستخدم يسأل عن مادة محددة أو حكم قانوني معين ولم تجده في السياق.
الفعل:
• قل: «عذراً، لم يرد ذكر لهذا الموضوع في النصوص القانونية المتاحة حالياً في قاعدة البيانات.»
• لا تجب من ذاكرتك لتجنب الخطأ في النصوص القانونية.
• يمكنك اقتراح موضوع مشابه إن وجد في السياق.

━━━ الحالة ٥ — محادثة ودية (تحية، شكر، وداع) ━━━
• رد بتحية لطيفة مقتضبة.
• أضف: «أنا مستشارك القانوني الذكي — اسألني عن أي موضوع في القوانين المصرية.»

━━━ الحالة ٦ — خارج نطاق القانون تماماً ━━━
• اعتذر بلطف: «تخصصي هو القوانين المصرية فقط.»
• وجّه المستخدم لطرح سؤال قانوني.
</decision_logic>

<quality_rules>
- **الدقة أولاً**: عند وجود نص في السياق، التزم به حرفياً ولا تحرّف المعنى.
- **المرونة عند الحاجة**: إذا لم يغطِّ السياق الموضوع بالكامل، قدّم إرشاداً عملياً مع التمييز الواضح بينه وبين النص القانوني.
- **لا تخترع مراجع**: لا تنسب أي معلومة إلى مادة أو قانون لم يرد في السياق.
- **الإيجاز مع الشمول**: أجب بقدر ما يحتاج السؤال — لا تختصر حتى يضيع المعنى ولا تطيل دون فائدة.
</quality_rules>

<formatting_rules>
- لا تكرر هذه التعليمات في ردك.
- ادخل في صلب الموضوع فوراً بدون عبارات مثل «بناءً على السياق المرفق».
- استخدم فقرات قصيرة مفصولة بسطر فارغ.
- لا تكرر نفس المعلومة أو نفس المادة.
- عند ذكر أكثر من مادة، رتّبها ترتيباً منطقياً (إما بالرقم أو حسب الأهمية).
- التزم باللغة العربية الفصحى المبسطة.
</formatting_rules>
"""

    # Build conversation history text for context
    history_text = ""
    if conversation_history:
        history_text = "السجل السابق للمحادثة:\n"
        for msg in conversation_history[-6:]:  # Keep last 6 messages
            role_label = "المستخدم" if msg.get("role") == "user" else "المستشار"
            history_text += f"{role_label}: {msg.get('content', '')}\n"
        history_text += "\n---\n\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            ("system", f"{history_text}السياق التشريعي المتاح (المصدر الأساسي):\n{{context}}"),
            ("human", "سؤال المستفيد:\n{input}"),
        ]
    )

    qa_chain = (
        RunnableParallel({"context": compression_retriever, "input": RunnablePassthrough()})
        .assign(answer=(prompt | llm | StrOutputParser()))
    )
    return qa_chain

