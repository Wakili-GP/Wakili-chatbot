
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

    # Embedding model — switch here to change embeddings globally
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    )
    embedding_cache_dir: str = os.path.join(base_dir, "embedding_cache")

    # Chroma dir — override with CHROMA_DIR env var, or auto-name after model
    chroma_dir: str = os.getenv("CHROMA_DIR", "") or os.path.join(
        base_dir,
        "chroma_db",
    )

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    # For faster responses, try: llama-3.1-8b-instant (2-3x faster, lower quality)
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    reranker_model_path: str = os.getenv("RERANKER_MODEL_PATH", "")
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: _parse_origins(os.getenv("CORS_ALLOWED_ORIGINS", "*"))
    )

    # Retrieval — balance quality vs speed
    semantic_k: int = int(os.getenv("SEMANTIC_K", "10"))
    bm25_k: int = int(os.getenv("BM25_K", "10"))
    meta_k: int = int(os.getenv("META_K", "8"))
    hybrid_top_k: int = int(os.getenv("HYBRID_TOP_K", "7"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "3"))

    # RRF weights
    beta_semantic: float = float(os.getenv("BETA_SEMANTIC", "0.60"))
    beta_bm25: float = float(os.getenv("BETA_BM25", "0.20"))
    beta_metadata: float = float(os.getenv("BETA_METADATA", "0.20"))

    # LLM
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.80"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "45"))

    # Chat history depth
    chat_history_turns: int = int(os.getenv("CHAT_HISTORY_TURNS", "3"))

    # Response cache (in-memory; avoids re-running RAG for identical queries)
    response_cache_maxsize: int = int(os.getenv("RESPONSE_CACHE_MAXSIZE", "128"))
    response_cache_ttl: int = int(os.getenv("RESPONSE_CACHE_TTL", "300"))  # seconds


settings = Settings()
