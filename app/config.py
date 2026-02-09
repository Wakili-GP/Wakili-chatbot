
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    chroma_dir: str = os.path.join(base_dir, "chroma_db")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    reranker_model_path: str = os.getenv("RERANKER_MODEL_PATH", "")

    semantic_k: int = int(os.getenv("SEMANTIC_K", "10"))
    bm25_k: int = int(os.getenv("BM25_K", "10"))
    meta_k: int = int(os.getenv("META_K", "10"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "12"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.85"))


settings = Settings()
