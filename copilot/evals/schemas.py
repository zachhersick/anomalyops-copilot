from pydantic import BaseModel


class RetrievalEvalCase(BaseModel):
    case_id: str
    query: str
    expected_source_paths: list[str]
    top_k: int
    
    
class RetrievalEvalResult(BaseModel):
    case_id: str
    query: str
    expected_source_paths: list[str]
    retrieved_source_paths: list[str]
    passed: bool
    
    
class RetrievalEvalReport(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    hit_rate: float
    results: list[RetrievalEvalResult]