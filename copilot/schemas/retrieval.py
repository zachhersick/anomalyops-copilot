from pydantic import BaseModel
from copilot.schemas.chunk import SourceChunk


class ScoredChunk(BaseModel):
    chunk: SourceChunk
    score: float