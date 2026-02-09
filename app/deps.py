
# ============================================
# file: app/deps.py
# ============================================
from __future__ import annotations

import threading
from typing import Any, Optional

from .config import settings
from .rag_pipeline import build_qa_chain

_lock = threading.RLock()
_chain: Optional[Any] = None


def get_chain():
    global _chain
    with _lock:
        if _chain is None:
            _chain = build_qa_chain(settings)
        return _chain


def reload_chain():
    global _chain
    with _lock:
        _chain = build_qa_chain(settings)
        return _chain


