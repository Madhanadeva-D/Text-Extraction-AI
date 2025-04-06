from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class LoadRequest(BaseModel):
    source_type: str  # "url", "image", "pdf"
    content: str  # URL or base64 encoded file content
    filename: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    score_threshold: Optional[float] = 0.7

class Response(BaseModel):
    status: str
    message: str
    chunks: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    relevant_documents: Optional[List[dict]] = None