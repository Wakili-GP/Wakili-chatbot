from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question in Arabic")
    session_id: str = Field(default="default", description="Unique session identifier")
    include_sources: bool = Field(default=True, description="Return retrieved source docs")
    eastern_arabic_numerals: bool = Field(
        default=False, description="Convert digits 0-9 to Eastern Arabic numerals"
    )


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
    sources: List[SourceDoc] = Field(default_factory=list)
    session_id: str
    raw: Dict[str, Any] = Field(default_factory=dict)
