from pydantic import BaseModel, Field
from copilot.schemas.answer import Citation


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, gt=0)
    min_score: float = Field(default=0.0, ge=0.0)
    show_context: bool = False
    
    
class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    refusal_reason: str | None = None