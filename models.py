from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    article_number: Optional[str] = None
    legal_nature: Optional[str] = None
    keywords: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ArticleItem(BaseModel):
    law_key: Optional[str] = None
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    original_text: Optional[str] = None
    simplified_summary: Optional[str] = None
    legal_nature: Optional[str] = None

class AskResponse(BaseModel):
    answer: Optional[str] = None
    articles: Optional[List[ArticleItem]] = None
    sources: List[SourceItem] = []
