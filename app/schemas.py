from pydantic import BaseModel, Field
from typing import List


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    system: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    preview: str


class SearchResponse(BaseModel):
    query: str
    system: str
    top_k: int
    results: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    documents_loaded: bool
    assets_available: bool
