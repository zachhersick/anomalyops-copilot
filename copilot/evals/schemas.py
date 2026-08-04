from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    
    
class RagEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_source_paths: list[str]
    expect_refusal: bool = False
    top_k: int = Field(default=3, gt=0)
    min_score: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_expected_sources(self):
        if (
            not self.expect_refusal
            and not self.expected_source_paths
        ):
            raise ValueError(
                "Supported RAG cases require expected source paths."
            )

        return self


class RagEvalResult(BaseModel):
    case_id: str
    status: Literal[
        "answered",
        "refused",
        "invalid",
    ]
    expected_source_paths: list[str]
    retrieved_source_paths: list[str]
    cited_source_paths: list[str]
    schema_valid: bool
    retrieval_hit: bool
    citations_valid: bool
    citation_hit: bool
    refusal_correct: bool
    passed: bool
    failure_reasons: list[str]


class RagEvalReport(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    supported_cases: int
    refusal_cases: int
    schema_validity_rate: float
    retrieval_hit_rate: float
    citation_validity_rate: float
    citation_hit_rate: float
    refusal_accuracy: float
    pass_rate: float
    results: list[RagEvalResult]