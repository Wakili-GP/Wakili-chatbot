from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    """Frontend requests a new session — optionally supplies a user_id."""
    user_id: Optional[str] = Field(default=None, description="Optional user identifier; one will be generated if absent")


class SessionResponse(BaseModel):
    """Returned to the frontend so it can attach session_id to every subsequent request."""
    session_id: str
    user_id: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question in Arabic")
    user_id: str = Field(..., min_length=1, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier for conversation tracking")
    include_sources: bool = Field(default=True, description="Return retrieved source docs")
    eastern_arabic_numerals: bool = Field(
        default=False, description="Convert digits 0-9 to Eastern Arabic numerals"
    )


class Message(BaseModel):
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., min_length=1, description="Message text")


class SourceDoc(BaseModel):
    article_id: Optional[str] = None
    article_number: Optional[str] = None
    law_name: Optional[str] = None
    legal_nature: Optional[str] = None
    keywords: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    page_content: str


class AskResponse(BaseModel):
    answer: str
    user_id: str
    session_id: Optional[str] = None
    sources: List[SourceDoc] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    user_id: str
    session_id: str
    history: List[Message] = Field(default_factory=list)


class ClearHistoryResponse(BaseModel):
    user_id: str
    session_id: str
    cleared: bool = True
