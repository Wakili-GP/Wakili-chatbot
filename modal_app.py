from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "wakili-api"

# ---------------------------------------------------------------------------
# Build-time function: pre-download ALL ML models into the image cache
# so they never need to be downloaded at runtime.
# ---------------------------------------------------------------------------
def prefetch_models():
    """Pre-download embeddings + reranker models at image build time."""
    import torch  # noqa: F401 – ensures torch is importable

    # 1. Embeddings model (~500 MB)
    from sentence_transformers import SentenceTransformer
    print("⏳ Pre-downloading embeddings model: Omartificial-Intelligence-Space/GATE-AraBert-v1")
    SentenceTransformer("Omartificial-Intelligence-Space/GATE-AraBert-v1")
    print("✓ Embeddings model cached")

    # 2. Reranker / cross-encoder model (~1.1 GB)
    from sentence_transformers import CrossEncoder
    print("⏳ Pre-downloading reranker model: BAAI/bge-reranker-v2-m3")
    CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("✓ Reranker model cached")


# ---------------------------------------------------------------------------
# Image: lean requirements + pre-fetched models + application code & data
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements-modal.txt")
    .run_function(prefetch_models)            # cache ML weights at build time
    .add_local_dir("app",       remote_path="/root/app")
    .add_local_dir("data",      remote_path="/root/data")
    .add_local_dir("chroma_db", remote_path="/root/chroma_db")  # pre-built vector DB
)

# Only add reranker tokenizer files if dir exists (weights come from cache)
if Path("reranker").exists():
    image = image.add_local_dir("reranker", remote_path="/root/reranker")

# prefetch_models already ran above; no additional run_function needed

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Serve the FastAPI app
# ---------------------------------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name("wakili-secrets")],
    timeout=300,              # 5 min per request
    scaledown_window=1800,    # keep container alive 30 min after last request
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")

    # Point to bundled reranker tokenizer (model weights loaded from HF cache)
    os.environ.setdefault("CHROMA_DIR", "/root/chroma_db")
    os.environ.setdefault("RERANKER_MODEL_PATH", "/root/reranker")
    os.environ.setdefault("CORS_ALLOWED_ORIGINS",
                          "https://www.wakili.me,https://wakili.me,http://localhost:3000")
    os.environ.setdefault("PRELOAD_CHAIN_ON_STARTUP", "true")
    os.environ.setdefault("HISTORY_BACKEND", "modal_dict")
    os.environ.setdefault("HISTORY_MODAL_DICT_NAME", "wakili-history")

    from app.main import app as fastapi_application

    return fastapi_application
