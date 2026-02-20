
# ============================================
# file: app/deps.py
# ============================================
from __future__ import annotations

from typing import Any, List, Optional

from .config import settings
from .rag_pipeline import build_qa_chain


def get_chain(conversation_history: List[dict] = None):
    """
    Build the QA chain with conversation history.
    Always rebuilds the chain to include current conversation context.
    
    Args:
        conversation_history: List of previous messages for context
    """
    # Always build/rebuild the chain with current conversation history
    # This ensures each request gets the proper context
    return build_qa_chain(settings, conversation_history=conversation_history)


def reload_chain():
    """Rebuild the QA chain from scratch without conversation history."""
    return build_qa_chain(settings)


