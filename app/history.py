# ============================================
# file: app/history.py
# ============================================
from __future__ import annotations

import threading
from typing import Dict, List

from .schemas import Message

_lock = threading.RLock()
_conversations: Dict[str, List[Message]] = {}


def get_history(session_id: str) -> List[Message]:
    """Retrieve conversation history for a session."""
    with _lock:
        return _conversations.get(session_id, [])


def add_to_history(session_id: str, user_msg: str, assistant_msg: str):
    """Add user and assistant messages to conversation history."""
    with _lock:
        if session_id not in _conversations:
            _conversations[session_id] = []
        _conversations[session_id].append(Message(role="user", content=user_msg))
        _conversations[session_id].append(Message(role="assistant", content=assistant_msg))


def clear_history(session_id: str):
    """Clear conversation history for a session."""
    with _lock:
        if session_id in _conversations:
            del _conversations[session_id]


def get_history_text(session_id: str, max_messages: int = 6) -> str:
    """Convert conversation history to formatted text for LLM context."""
    history = get_history(session_id)
    if not history:
        return ""
    
    # Keep only last max_messages messages
    recent = history[-max_messages:]
    
    history_text = "السجل السابق للمحادثة:\n"
    for msg in recent:
        role_label = "المستخدم" if msg.role == "user" else "المستشار"
        history_text += f"{role_label}: {msg.content}\n"
    
    return history_text + "\n---\n\n"
