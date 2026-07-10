from pydantic import BaseModel


class RetrievalEvalCase(BaseModel):
    case_id: str
    query: str
    expected_source_paths: list[str]
    top_k: int