from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: int
    source_path: str
    start_line: int
    end_line: int
    
    
class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    refusal_reason: str | None = None