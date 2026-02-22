# ============================================
# file: app/history.py
# ============================================
from __future__ import annotations

import os
import threading
from typing import Dict, List

from .schemas import Message

_lock = threading.RLock()
_conversations: Dict[str, List[Message]] = {}

_history_backend = os.getenv("HISTORY_BACKEND", "memory").strip().lower()
_modal_dict_name = os.getenv("HISTORY_MODAL_DICT_NAME", "wakili-history")
_shared_store = None

if _history_backend == "modal_dict":
    try:
        import modal

        _shared_store = modal.Dict.from_name(_modal_dict_name, create_if_missing=True)
    except Exception:
        _shared_store = None
        _history_backend = "memory"


def _conversation_key(user_id: str, session_id: str) -> str:
    return f"{user_id}::{session_id}"


def get_history(user_id: str, session_id: str) -> List[Message]:
    """Retrieve conversation history for a user session."""
    key = _conversation_key(user_id, session_id)
    if _history_backend == "modal_dict" and _shared_store is not None:
        raw_messages = _shared_store.get(key, [])
        return [Message(**msg) for msg in raw_messages]

    with _lock:
        return _conversations.get(key, [])


def add_to_history(user_id: str, session_id: str, user_msg: str, assistant_msg: str):
    """Add user and assistant messages to conversation history."""
    key = _conversation_key(user_id, session_id)
    if _history_backend == "modal_dict" and _shared_store is not None:
        history = _shared_store.get(key, [])
        history.append(Message(role="user", content=user_msg).model_dump())
        history.append(Message(role="assistant", content=assistant_msg).model_dump())
        _shared_store[key] = history
        return

    with _lock:
        if key not in _conversations:
            _conversations[key] = []
        _conversations[key].append(Message(role="user", content=user_msg))
        _conversations[key].append(Message(role="assistant", content=assistant_msg))


def clear_history(user_id: str, session_id: str):
    """Clear conversation history for a user session."""
    key = _conversation_key(user_id, session_id)
    if _history_backend == "modal_dict" and _shared_store is not None:
        try:
            _shared_store.pop(key)
        except KeyError:
            pass
        return

    with _lock:
        if key in _conversations:
            del _conversations[key]


def get_history_text(user_id: str, session_id: str, max_messages: int = 6) -> str:
    """Convert conversation history to formatted text for LLM context."""
    history = get_history(user_id, session_id)
    if not history:
        return ""
    
    # Keep only last max_messages messages
    recent = history[-max_messages:]
    
    history_text = "السجل السابق للمحادثة:\n"
    for msg in recent:
        role_label = "المستخدم" if msg.role == "user" else "المستشار"
        history_text += f"{role_label}: {msg.content}\n"
    
    return history_text + "\n---\n\n"
