
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_origins(value: str) -> List[str]:
    if not value:
        return ["*"]
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass(frozen=True)
class Settings:
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    chroma_dir: str = os.path.join(base_dir, "chroma_db")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    reranker_model_path: str = os.getenv("RERANKER_MODEL_PATH", "")
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: _parse_origins(os.getenv("CORS_ALLOWED_ORIGINS", "*"))
    )

    semantic_k: int = int(os.getenv("SEMANTIC_K", "6"))
    bm25_k: int = int(os.getenv("BM25_K", "6"))
    meta_k: int = int(os.getenv("META_K", "6"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "8"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "3"))

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.85"))


settings = Settings()
