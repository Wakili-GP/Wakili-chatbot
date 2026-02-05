# -*- coding: utf-8 -*-
import os
import sys
import json
from dotenv import load_dotenv
import streamlit as st
import logging
import warnings

# Suppress progress bars from transformers/tqdm
os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'
warnings.filterwarnings('ignore')

# 1. Loaders & Splitters
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

# 2. Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Reranker Imports
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 4. LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 🎨 UI SETUP (CSS FOR ARABIC & RTL)
# ==========================================
st.set_page_config(page_title="المساعد القانوني", page_icon="⚖️")

# This CSS block fixes the "001" number issue and right alignment
st.markdown("""
<style>
    /* Force the main app container to be Right-to-Left */
    .stApp {
        direction: rtl;
        text-align: right;
    }
    
    /* Fix input fields to type from right */
    .stTextInput input {
        direction: rtl;
        text-align: right;
    }

    /* Fix chat messages alignment */
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    
    /* Ensure proper paragraph spacing */
    .stMarkdown p {
        margin: 0.5em 0 !important;
        line-height: 1.6;
        word-spacing: 0.1em;
    }
    
    /* Ensure numbers display correctly in RTL */
    p, div, span, label {
        unicode-bidi: embed;
        direction: inherit;
        white-space: normal;
        word-wrap: break-word;
    }
    
    /* Force all content to respect RTL */
    * {
        direction: rtl !important;
    }
    
    /* Preserve line breaks and spacing */
    .stMarkdown pre {
        direction: rtl;
        text-align: right;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Hide the "Deploy" button and standard menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# Put this at the top of your code
def convert_to_eastern_arabic(text):
    """Converts 0123456789 to ٠١٢٣٤٥٦٧٨٩"""
    if not isinstance(text, str):
        return text 
    western_numerals = '0123456789'
    eastern_numerals = '٠١٢٣٤٥٦٧٨٩'
    translation_table = str.maketrans(western_numerals, eastern_numerals)
    return text.translate(translation_table)

st.title("⚖️ المساعد القانوني الذكي (دستور مصر)")

# ==========================================
# 🚀 CACHED RESOURCE LOADING (THE FIX)
# ==========================================
# This decorator tells Streamlit: "Run this ONCE and save the result."
@st.cache_resource
def initialize_rag_pipeline():
    print("🔄 Initializing system...")
    print("📥 Loading data...")
    
    # 1. Load JSON
    json_path = "Egyptian_Constitution_legalnature_only.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a mapping of article numbers for cross-reference lookup
    article_map = {str(item['article_number']): item for item in data}

    docs = []
    for item in data:
        # Build cross-reference section
        cross_ref_text = ""
        if item.get('cross_references') and len(item['cross_references']) > 0:
            cross_ref_text = "\nالمواد ذات الصلة (المراجع المتقاطعة): " + ", ".join(
                [f"المادة {ref}" for ref in item['cross_references']]
            )
        
        # Construct content
        page_content = f"""
        رقم المادة: {item['article_number']}
        النص الأصلي: {item['original_text']}
        الشرح المبسط: {item['simplified_summary']}{cross_ref_text}
        """
        metadata = {
            "article_id": item['article_id'],
            "article_number": str(item['article_number']),
            "legal_nature": item['legal_nature'],
            "keywords": ", ".join(item['keywords']),
            "part": item.get('part (Bab)', ''),
            "chapter": item.get('chapter (Fasl)', ''),
            "cross_references": ", ".join([str(ref) for ref in item.get('cross_references', [])])  # Convert list to string
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"✅ Loaded {len(docs)} constitutional articles")

    # 2. Embeddings
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
    )
    print("✅ Embeddings model ready")

    # 3. No splitting - keep articles as complete units
    chunks = docs

    # 4. Vector Store
    print("Building vector database...")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db"
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    print("✅ Vector database ready")

    # 5. Create BM25 Keyword Retriever
    class BM25Retriever(BaseRetriever):
        """BM25-based keyword retriever for constitutional articles"""
        corpus_docs: List[Document]
        bm25: BM25Okapi = None
        k: int = 15
        
        class Config:
            arbitrary_types_allowed = True
        
        def __init__(self, **data):
            super().__init__(**data)
            # Tokenize corpus for BM25
            tokenized_corpus = [doc.page_content.split() for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            # Tokenize query
            tokenized_query = query.split()
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            # Get top k indices
            top_indices = np.argsort(scores)[::-1][:self.k]
            # Return documents
            return [self.corpus_docs[i] for i in top_indices if scores[i] > 0]
        
        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)
    
    bm25_retriever = BM25Retriever(corpus_docs=docs, k=15)
    print("✅ BM25 keyword retriever ready")

    # 6. Create Metadata Filter Retriever
    class MetadataFilterRetriever(BaseRetriever):
        """Metadata-based filtering retriever"""
        corpus_docs: List[Document]
        k: int = 15
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            query_lower = query.lower()
            scored_docs = []
            
            for doc in self.corpus_docs:
                score = 0
                # Match keywords
                keywords = doc.metadata.get('keywords', '').lower()
                if any(word in keywords for word in query_lower.split()):
                    score += 3
                
                # Match legal nature
                legal_nature = doc.metadata.get('legal_nature', '').lower()
                if any(word in legal_nature for word in query_lower.split()):
                    score += 2
                
                # Match part/chapter
                part = doc.metadata.get('part', '').lower()
                chapter = doc.metadata.get('chapter', '').lower()
                if any(word in part or word in chapter for word in query_lower.split()):
                    score += 1
                
                # Match in content
                if any(word in doc.page_content.lower() for word in query_lower.split()):
                    score += 1
                
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:self.k]]
        
        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)
    
    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=15)
    print("✅ Metadata filter retriever ready")

    # 7. Create Hybrid RRF Retriever
    class HybridRRFRetriever(BaseRetriever):
        """Combines semantic, BM25, and metadata retrievers using Reciprocal Rank Fusion"""
        semantic_retriever: BaseRetriever
        bm25_retriever: BM25Retriever
        metadata_retriever: MetadataFilterRetriever
        beta_semantic: float = 0.6  # Weight for semantic search
        beta_keyword: float = 0.2   # Weight for BM25 keyword search
        beta_metadata: float = 0.2  # Weight for metadata filtering
        k: int = 60  # RRF constant (typically 60)
        top_k: int = 15
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            # Get results from all three retrievers
            semantic_docs = self.semantic_retriever.invoke(query)
            bm25_docs = self.bm25_retriever.invoke(query)
            metadata_docs = self.metadata_retriever.invoke(query)
            
            # Apply Reciprocal Rank Fusion
            rrf_scores = {}
            
            # Process semantic results
            for rank, doc in enumerate(semantic_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_semantic / (self.k + rank)
            
            # Process BM25 results
            for rank, doc in enumerate(bm25_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_keyword / (self.k + rank)
            
            # Process metadata results
            for rank, doc in enumerate(metadata_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_metadata / (self.k + rank)
            
            # Create document lookup
            all_docs = {}
            for doc in semantic_docs + bm25_docs + metadata_docs:
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
            
            # Sort by RRF score
            sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top k documents
            result_docs = []
            for doc_id, score in sorted_doc_ids[:self.top_k]:
                if doc_id in all_docs:
                    result_docs.append(all_docs[doc_id])
            
            return result_docs
        
        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)
    
    # Create hybrid retriever with tuned beta weights
    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=0.5,   # Semantic search gets highest weight (most reliable)
        beta_keyword=0.3,    # BM25 keyword search (good for exact term matches)
        beta_metadata=0.2,   # Metadata filtering (supporting role)
        k=60,
        top_k=20
    )
    print("✅ Hybrid RRF retriever ready with β weights: semantic=0.5, keyword=0.3, metadata=0.2")

    # 8. Create Cross-Reference Enhanced Retriever
    class CrossReferenceRetriever(BaseRetriever):
        """Enhances retrieval by automatically fetching cross-referenced articles"""
        base_retriever: BaseRetriever
        article_map: dict
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            # Get initial results
            initial_docs = self.base_retriever.invoke(query)
            
            # Collect all related article numbers
            all_article_numbers = set()
            for doc in initial_docs:
                if 'article_number' in doc.metadata:
                    all_article_numbers.add(doc.metadata['article_number'])
                # Parse cross_references (now stored as comma-separated string)
                cross_refs_str = doc.metadata.get('cross_references', '')
                if cross_refs_str:
                    cross_refs = [ref.strip() for ref in cross_refs_str.split(',')]
                    for ref in cross_refs:
                        if ref:  # Skip empty strings
                            all_article_numbers.add(str(ref))
            
            # Build enhanced document list
            enhanced_docs = []
            seen_numbers = set()
            
            # Add initially retrieved documents
            for doc in initial_docs:
                enhanced_docs.append(doc)
                seen_numbers.add(doc.metadata.get('article_number'))
            
            # Add cross-referenced articles not yet retrieved
            for article_num in all_article_numbers:
                if article_num not in seen_numbers and article_num in self.article_map:
                    article_data = self.article_map[article_num]
                    cross_ref_text = ""
                    if article_data.get('cross_references'):
                        cross_ref_text = "\nالمواد ذات الصلة: " + ", ".join(
                            [f"المادة {ref}" for ref in article_data['cross_references']]
                        )
                    
                    page_content = f"""
                    رقم المادة: {article_data['article_number']}
                    النص الأصلي: {article_data['original_text']}
                    الشرح المبسط: {article_data['simplified_summary']}{cross_ref_text}
                    """
                    
                    enhanced_doc = Document(
                        page_content=page_content,
                        metadata={
                            "article_id": article_data['article_id'],
                            "article_number": str(article_data['article_number']),
                            "legal_nature": article_data['legal_nature'],
                            "keywords": ", ".join(article_data['keywords']),
                            "cross_references": ", ".join([str(ref) for ref in article_data.get('cross_references', [])])
                        }
                    )
                    enhanced_docs.append(enhanced_doc)
                    seen_numbers.add(article_num)
            
            return enhanced_docs
        
        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)
    
    cross_ref_retriever = CrossReferenceRetriever(
        base_retriever=hybrid_retriever,
        article_map=article_map
    )
    print("✅ Cross-reference retriever ready (using hybrid RRF base)")

    # 9. Reranker
    print("Loading reranker model...")
    local_model_path = r"D:\FOE\Senior 2\Graduation Project\Chatbot_me\reranker"
    
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Reranker path not found: {local_model_path}")

    model = HuggingFaceCrossEncoder(model_name=local_model_path)
    compressor = CrossEncoderReranker(model=model, top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=cross_ref_retriever
    )
    print("✅ Reranker model ready")

    # 7. LLM - Balanced for consistency with slight creativity
    # 7. LLM Configuration
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,       # Slightly increased to allow helpful general advice
        model_kwargs={"top_p": 0.9}
    )

# ==================================================
    # 🛠️ THE FIX: SEPARATE SYSTEM INSTRUCTIONS FROM USER INPUT
    # ==================================================
    
# ==================================================
    # 🧠 PROMPT ENGINEERING: DECISION TREE LOGIC
    # ==================================================
    
    system_instructions = """
    <role>
    أنت "المساعد القانوني الذكي"، خبير متخصص في الدستور المصري والقوانين الإجرائية.
    مهمتك: تقديم إجابات دقيقة بناءً على "السياق التشريعي" المرفق أولاً، أو تقديم نصائح إجرائية عامة عند الضرورة.
    </role>

    <decision_logic>
    عليك تحليل "سؤال المستخدم" و"السياق التشريعي" وتصنيف الحالة واختيار الرد المناسب بناءً على القواعد التالية بدقة:

    🔴 الحالة الأولى: (الإجابة موجودة في السياق التشريعي)
    الشرط: إذا وجدت معلومات داخل "السياق التشريعي المتاح" تجيب على السؤال.
    الفعل:
    1. استخرج الإجابة من السياق فقط.
    2. ابدأ الإجابة مباشرة دون مقدمات.
    3. يجب توثيق الإجابة برقم المادة (مثال: "نصت المادة (50) على...").
    4. توقف هنا. لا تضف أي معلومات خارجية.

    🟡 الحالة الثانية: (السياق فارغ/غير مفيد + السؤال إجرائي/عملي)
    الشرط: إذا لم تجد الإجابة في السياق، وكان السؤال عن إجراءات عملية (مثل: حادث، سرقة، طلاق، تحرير محضر، تعامل مع الشرطة).
    الفعل:
    1. تجاهل السياق الفارغ.
    2. استخدم معرفتك العامة بالقانون المصري.
    3. ابدأ وجوباً بعبارة: "بناءً على الإجراءات القانونية العامة في مصر (وليس نصاً دستورياً محدداً):"
    4. قدم الخطوات في نقاط مرقمة واضحة ومختصرة (1، 2، 3).
    5. تحذير: لا تذكر أرقام مواد قانونية (لا تخترع أرقام مواد).

    🔵 الحالة الثالثة: (السياق فارغ + السؤال عن نص دستوري محدد)
    الشرط: إذا سأل عن (مجلس الشعب، الشورى، مادة محددة) ولم تجدها في السياق.
    الفعل:
    1. قل بوضوح: "عذراً، لم يرد ذكر لهذا الموضوع في المواد الدستورية التي تم استرجاعها في السياق الحالي."
    2. لا تحاول الإجابة من ذاكرتك لكي لا تخطئ في النصوص الدستورية الحساسة.

    🟢 الحالة الرابعة: (محادثة ودية)
    الشرط: تحية، شكر، أو "كيف حالك".
    الفعل: رد بتحية مهذبة جداً ومقتضبة، ثم قل: "أنا جاهز للإجابة على استفساراتك القانونية."

    ⚫ الحالة الخامسة: (خارج النطاق تماماً)
    الشرط: طبخ، رياضة، برمجة، أو أي موضوع غير قانوني.
    الفعل: اعتذر بلطف ووجه المستخدم للسؤال في القانون.
    </decision_logic>

    <formatting_rules>
    - لا تكرر هذه التعليمات في ردك.
    - استخدم فقرات قصيرة واترك سطراً فارغاً بينها.
    - لا تستخدم عبارات مثل "بناء على السياق المرفق" في بداية الجملة، بل ادخل في صلب الموضوع فوراً.
    - التزم باللغة العربية الفصحى المبسطة والرصينة.
    </formatting_rules>
    """

    # We use .from_messages to strictly separate instructions from data
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("system", "السياق التشريعي المتاح (المصدر الأساسي):\n{context}"), 
        ("human", "سؤال المستفيد:\n{input}")
    ])
    
    # 9. Build Chain with RunnableParallel (returns both context and answer)
    qa_chain = (
        RunnableParallel({
            "context": compression_retriever, 
            "input": RunnablePassthrough()
        })
        .assign(answer=(
            prompt 
            | llm 
            | StrOutputParser()
        ))
    )
    
    print("✅ System ready to use!")
    return qa_chain

# ==========================================
# ⚡ MAIN EXECUTION
# ==========================================

try:
    # Only need the chain now - it handles all retrieval internally
    qa_chain = initialize_rag_pipeline()
    
except Exception as e:
    st.error(f"Critical Error loading application: {e}")
    st.stop()

# ==========================================
# 💬 CHAT LOOP
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History (with Eastern Arabic numerals)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Convert to Eastern Arabic when displaying from history
        st.markdown(convert_to_eastern_arabic(message["content"]))

# Handle New User Input
if prompt_input := st.chat_input("اكتب سؤالك القانوني هنا..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("جاري التحليل القانوني..."):
            try:
                # Invoke chain ONCE - returns Dict with 'context', 'input', and 'answer'
                result = qa_chain.invoke(prompt_input)
                
                # Extract answer and context from result
                response_text = result["answer"]
                source_docs = result["context"]  # Context is already in the result!

                # Display Answer
                response_text_arabic = convert_to_eastern_arabic(response_text)
                st.markdown(response_text_arabic)
                
                # Display Sources
                if source_docs and len(source_docs) > 0:
                    print(f"✅ Found {len(source_docs)} documents")
                    # Deduplicate documents by article_number
                    seen_articles = set()
                    unique_docs = []
                    
                    for doc in source_docs:
                        article_num = str(doc.metadata.get('article_number', '')).strip()
                        if article_num and article_num not in seen_articles:
                            seen_articles.add(article_num)
                            unique_docs.append(doc)
                    
                    st.markdown("---")  # Separator before sources
                    
                    if unique_docs:
                        with st.expander(f"📚 المصادر المستخدمة ({len(unique_docs)} مادة)"):
                            st.markdown("### المواد الدستورية المستخدمة في التحليل:")
                            st.markdown("---")
                            
                            for idx, doc in enumerate(unique_docs, 1):
                                article_num = str(doc.metadata.get('article_number', '')).strip()
                                legal_nature = doc.metadata.get('legal_nature', '')
                                
                                if article_num:
                                    st.markdown(f"**المادة رقم {convert_to_eastern_arabic(article_num)}**")
                                    if legal_nature:
                                        st.markdown(f"*الطبيعة القانونية: {legal_nature}*")
                                    
                                    # Display article content
                                    content_lines = doc.page_content.strip().split('\n')
                                    for line in content_lines:
                                        line = line.strip()
                                        if line:
                                            st.markdown(convert_to_eastern_arabic(line))
                                    
                                    st.markdown("---")
                    else:
                        st.info("📌 لم يتم العثور على مصادر")
                else:
                    st.info("📌 لم يتم العثور على مصادر")
                
                # Persist the raw answer to avoid double conversion glitches on rerun
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"حدث خطأ: {e}")