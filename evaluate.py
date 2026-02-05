# -*- coding: utf-8 -*-
"""
RAGAS Evaluation Script for Constitutional Legal Assistant
Evaluates: faithfulness, answer_relevancy, context_precision, context_recall
"""

import os
import json
from dotenv import load_dotenv
import logging
import warnings

# Suppress progress bars
os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'
warnings.filterwarnings('ignore')

# Core imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

# Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Reranker
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Evaluation
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 🚀 RAG PIPELINE INITIALIZATION
# ==========================================

def initialize_rag_pipeline():
    """Initialize the RAG pipeline for constitutional legal questions"""
    print("🔄 Initializing RAG pipeline...")
    print("📥 Loading data...")
    
    # 1. Load JSON
    json_path = "Egyptian_Constitution_legalnature_only.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create article mapping for cross-references
    article_map = {str(item['article_number']): item for item in data}

    docs = []
    for item in data:
        # Build cross-reference section
        cross_ref_text = ""
        if item.get('cross_references') and len(item['cross_references']) > 0:
            cross_ref_text = "\nالمواد ذات الصلة (المراجع المتقاطعة): " + ", ".join(
                [f"المادة {ref}" for ref in item['cross_references']]
            )
        
        # Construct document content
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
            "cross_references": ", ".join([str(ref) for ref in item.get('cross_references', [])])
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"✅ Loaded {len(docs)} constitutional articles")

    # 2. Embeddings
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
    )
    print("✅ Embeddings ready")

    # 3. Vector Store
    print("Building vector database...")
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chroma_db"
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    print("✅ Vector database ready")

    # 4. BM25 Keyword Retriever
    class BM25Retriever(BaseRetriever):
        """BM25-based keyword retriever"""
        corpus_docs: List[Document]
        bm25: BM25Okapi = None
        k: int = 15
        
        class Config:
            arbitrary_types_allowed = True
        
        def __init__(self, **data):
            super().__init__(**data)
            tokenized_corpus = [doc.page_content.split() for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:self.k]
            return [self.corpus_docs[i] for i in top_indices if scores[i] > 0]
        
        async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return self._get_relevant_documents(query)
    
    bm25_retriever = BM25Retriever(corpus_docs=docs, k=15)
    print("✅ BM25 retriever ready")

    # 5. Metadata Filter Retriever
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
                keywords = doc.metadata.get('keywords', '').lower()
                if any(word in keywords for word in query_lower.split()):
                    score += 3
                
                legal_nature = doc.metadata.get('legal_nature', '').lower()
                if any(word in legal_nature for word in query_lower.split()):
                    score += 2
                
                part = doc.metadata.get('part', '').lower()
                chapter = doc.metadata.get('chapter', '').lower()
                if any(word in part or word in chapter for word in query_lower.split()):
                    score += 1
                
                if any(word in doc.page_content.lower() for word in query_lower.split()):
                    score += 1
                
                if score > 0:
                    scored_docs.append((doc, score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:self.k]]
        
        async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return self._get_relevant_documents(query)
    
    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=15)
    print("✅ Metadata retriever ready")

    # 6. Hybrid RRF Retriever
    class HybridRRFRetriever(BaseRetriever):
        """Combines semantic, BM25, and metadata using Reciprocal Rank Fusion"""
        semantic_retriever: BaseRetriever
        bm25_retriever: BM25Retriever
        metadata_retriever: MetadataFilterRetriever
        beta_semantic: float = 0.5
        beta_keyword: float = 0.3
        beta_metadata: float = 0.2
        k: int = 60
        top_k: int = 15
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            semantic_docs = self.semantic_retriever.invoke(query)
            bm25_docs = self.bm25_retriever.invoke(query)
            metadata_docs = self.metadata_retriever.invoke(query)
            
            rrf_scores = {}
            
            for rank, doc in enumerate(semantic_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_semantic / (self.k + rank)
            
            for rank, doc in enumerate(bm25_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_keyword / (self.k + rank)
            
            for rank, doc in enumerate(metadata_docs, start=1):
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.beta_metadata / (self.k + rank)
            
            all_docs = {}
            for doc in semantic_docs + bm25_docs + metadata_docs:
                doc_id = doc.metadata.get('article_number', str(hash(doc.page_content)))
                if doc_id not in all_docs:
                    all_docs[doc_id] = doc
            
            sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            result_docs = []
            for doc_id, score in sorted_doc_ids[:self.top_k]:
                if doc_id in all_docs:
                    result_docs.append(all_docs[doc_id])
            
            return result_docs
        
        async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return self._get_relevant_documents(query)
    
    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=0.5,
        beta_keyword=0.3,
        beta_metadata=0.2,
        k=60,
        top_k=20
    )
    print("✅ Hybrid RRF retriever ready (β: semantic=0.5, keyword=0.3, metadata=0.2)")

    # 7. Cross-Reference Retriever
    class CrossReferenceRetriever(BaseRetriever):
        """Enhances retrieval by fetching cross-referenced articles"""
        base_retriever: BaseRetriever
        article_map: dict
        
        class Config:
            arbitrary_types_allowed = True
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            initial_docs = self.base_retriever.invoke(query)
            
            all_article_numbers = set()
            for doc in initial_docs:
                if 'article_number' in doc.metadata:
                    all_article_numbers.add(doc.metadata['article_number'])
                cross_refs_str = doc.metadata.get('cross_references', '')
                if cross_refs_str:
                    cross_refs = [ref.strip() for ref in cross_refs_str.split(',')]
                    for ref in cross_refs:
                        if ref:
                            all_article_numbers.add(str(ref))
            
            enhanced_docs = []
            seen_numbers = set()
            
            for doc in initial_docs:
                enhanced_docs.append(doc)
                seen_numbers.add(doc.metadata.get('article_number'))
            
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
        
        async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return self._get_relevant_documents(query)
    
    cross_ref_retriever = CrossReferenceRetriever(
        base_retriever=hybrid_retriever,
        article_map=article_map
    )
    print("✅ Cross-reference retriever ready")

    # 8. Reranker
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
    print("✅ Reranker ready (top_n=5)")

    # 9. LLM Configuration
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        model_kwargs={"top_p": 0.9}
    )

    # 10. Prompt Template
    system_instructions = """
    <role>
    أنت "المساعد القانوني الذكي"، خبير متخصص في الدستور المصري والقوانين الإجرائية.
    مهمتك: تقديم إجابات دقيقة بناءً على "السياق التشريعي" المرفق أولاً، أو تقديم نصائح إجرائية عامة عند الضرورة.
    </role>

    <decision_logic>
    عليك تحليل "سؤال المستخدم" و"السياق التشريعي" وتصنيف الحالة واختيار الرد المناسب:

    🔴 الحالة الأولى: (الإجابة موجودة في السياق التشريعي)
    - استخرج الإجابة من السياق فقط
    - ابدأ الإجابة مباشرة دون مقدمات
    - وثق الإجابة برقم المادة
    - توقف، لا تضف معلومات خارجية

    🟡 الحالة الثانية: (السياق فارغ + السؤال إجرائي/عملي)
    - استخدم معرفتك العامة بالقانون المصري
    - ابدأ بـ: "بناءً على الإجراءات القانونية العامة في مصر:"
    - قدم الخطوات في نقاط مرقمة

    🔵 الحالة الثالثة: (السياق فارغ + سؤال دستوري)
    - قل: "عذراً، لم يرد ذكر لهذا في المواد المسترجاعة"
    - لا تخترع نصوصاً دستورية

    🟢 الحالة الرابعة: (تحية/شكر)
    - رد بتحية مهذبة مختصرة

    ⚫ الحالة الخامسة: (خارج النطاق)
    - اعتذر بلطف ووجه للقانون
    </decision_logic>

    <formatting_rules>
    - استخدم فقرات قصيرة واترك سطراً فارغاً بينها
    - التزم باللغة العربية الفصحى المبسطة
    </formatting_rules>
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("system", "السياق التشريعي المتاح:\n{context}"), 
        ("human", "السؤال:\n{input}")
    ])
    
    # 11. Build QA Chain
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
    
    print("✅ RAG pipeline initialized!\n")
    return qa_chain

# ==========================================
# 📊 RAGAS EVALUATION
# ==========================================

def run_evaluation(test_file="test_dataset.json", output_file="evaluation_results.json"):
    """Run RAGAS evaluation on test dataset"""
    
    print("\n" + "="*60)
    print("📊 RAGAS EVALUATION")
    print("="*60)
    
    # Load test dataset
    print(f"\n📂 Loading test dataset: {test_file}")
    with open(test_file, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    print(f"✅ Loaded {len(test_questions)} test questions")
    
    # Initialize RAG pipeline
    print("\n📥 Initializing RAG pipeline...")
    qa_chain = initialize_rag_pipeline()
    
    # Generate answers
    print("\n🤖 Generating answers for evaluation...")
    results = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for idx, item in enumerate(test_questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        print(f"  [{idx}/{len(test_questions)}] Processing question {idx}...")
        
        try:
            result = qa_chain.invoke(question)
            answer = result["answer"]
            contexts = [doc.page_content for doc in result["context"]]
            
            results["question"].append(question)
            results["answer"].append(answer)
            results["contexts"].append(contexts)
            results["ground_truth"].append(ground_truth)
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:100]}")
            results["question"].append(question)
            results["answer"].append("Error generating answer")
            results["contexts"].append([])
            results["ground_truth"].append(ground_truth)
    
    # Run Ragas evaluation
    print("\n⚙️ Running RAGAS metrics...")
    dataset = Dataset.from_dict(results)
    
    # Configure evaluation LLM (same as main app)
    print("  📌 Using Groq (llama-3.1-8b-instant, temp=0.3, top_p=0.9)")
    evaluator_llm = LangchainLLMWrapper(ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        model_kwargs={"top_p": 0.9},
        max_retries=2
    ))
    
    # Configure evaluation embeddings (same as main app)
    print("  📌 Using HuggingFace (Omartificial-Intelligence-Space/GATE-AraBert-v1)")
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
    ))
    
    try:
        import time
        print("\n  ⏳ Evaluating each question separately with all metrics...")
        print("  ⚠️ This will take ~10-15 minutes (120 sec delay between questions)\n")
        
        # Evaluate each question separately to see results immediately
        all_scores = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": []
        }
        
        for q_idx in range(len(results["question"])):
            print(f"\n  📋 Question {q_idx + 1}/{len(results['question'])}: {results['question'][q_idx][:60]}...")
            
            # Create single-question dataset
            single_q_data = {
                "question": [results["question"][q_idx]],
                "answer": [results["answer"][q_idx]],
                "contexts": [results["contexts"][q_idx]],
                "ground_truth": [results["ground_truth"][q_idx]]
            }
            single_dataset = Dataset.from_dict(single_q_data)
            
            # Evaluate all metrics for this question
            try:
                q_result = evaluate(
                    single_dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    raise_exceptions=False
                )
                
                # Extract scores (handle if they're lists or single values)
                def get_score(value):
                    if isinstance(value, list):
                        return value[0] if len(value) > 0 else 0.0
                    return float(value) if value is not None else 0.0
                
                f_score = get_score(q_result['faithfulness'])
                a_score = get_score(q_result['answer_relevancy'])
                cp_score = get_score(q_result['context_precision'])
                cr_score = get_score(q_result['context_recall'])
                
                # Display scores for this question
                print(f"     Faithfulness       : {f_score:.4f}")
                print(f"     Answer Relevancy   : {a_score:.4f}")
                print(f"     Context Precision  : {cp_score:.4f}")
                print(f"     Context Recall     : {cr_score:.4f}")
                
                all_scores["faithfulness"].append(f_score)
                all_scores["answer_relevancy"].append(a_score)
                all_scores["context_precision"].append(cp_score)
                all_scores["context_recall"].append(cr_score)
                
            except Exception as e:
                print(f"     ❌ Error evaluating this question: {str(e)[:80]}")
                all_scores["faithfulness"].append(0.0)
                all_scores["answer_relevancy"].append(0.0)
                all_scores["context_precision"].append(0.0)
                all_scores["context_recall"].append(0.0)
            
            # Wait between questions to avoid rate limits
            if q_idx < len(results["question"]) - 1:
                print(f"\n     ⏳ Waiting 120 seconds (2 min) before next question...")
                time.sleep(120)
        
        # Calculate average scores
        eval_results = {
            "faithfulness": sum(all_scores["faithfulness"]) / len(all_scores["faithfulness"]) if all_scores["faithfulness"] else 0.0,
            "answer_relevancy": sum(all_scores["answer_relevancy"]) / len(all_scores["answer_relevancy"]) if all_scores["answer_relevancy"] else 0.0,
            "context_precision": sum(all_scores["context_precision"]) / len(all_scores["context_precision"]) if all_scores["context_precision"] else 0.0,
            "context_recall": sum(all_scores["context_recall"]) / len(all_scores["context_recall"]) if all_scores["context_recall"] else 0.0
        }
        
        # Display results
        print("\n" + "="*60)
        print("📈 EVALUATION RESULTS")
        print("="*60)
        
        for metric, score in eval_results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric:28s}: {score:.4f}")
        
        # Save results to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            results_dict = {
                "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                           for k, v in eval_results.items()},
                "test_samples": len(dataset),
                "test_file": test_file
            }
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print("="*60 + "\n")
        
        return eval_results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("\n⚠️ Make sure:")
        print("   1. GROQ_API_KEY is set in .env")
        print("   2. You have valid Groq API credits")
        print("   3. Internet connection is available")
        return None

# ==========================================
# 🎯 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys
    
    test_file = "test_dataset.json"
    output_file = "evaluation_results.json"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("\n" + "="*60)
    print("🚀 Constitutional Legal Assistant - RAGAS Evaluation")
    print("="*60)
    
    run_evaluation(test_file, output_file)
