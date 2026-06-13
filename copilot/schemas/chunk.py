from pydantic import BaseModel


class SourceChunk(BaseModel):
    chunk_id: str
    source_id: str
    project_name: str
    source_type: str
    source_path: str
    chunk_index: int
    content: str
    start_line: int
    end_line: int