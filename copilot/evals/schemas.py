from typing import Literal

from pydantic import BaseModel, Field, model_validator

from copilot.schemas.triage import TriageRequest


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
    
    
TriageStatus = Literal[
    "completed",
    "no_alerts",
    "incomplete_data",
    "refused",
]


class TriageEvalExpectedFinding(BaseModel):
    severity: Literal[
        "critical",
        "warning",
        "unknown",
    ]
    machine_id: int = Field(gt=0)
    sensor: str = Field(min_length=1)
    anomaly_type: str | None = None


class TriageEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    request: TriageRequest
    expected_status: TriageStatus | None = None
    expected_findings: list[
        TriageEvalExpectedFinding
    ] = Field(default_factory=list)
    expected_finding_count: int | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_expected_count(self):
        if (
            self.expected_finding_count
            is not None
            and self.expected_finding_count
            > self.request.max_events
        ):
            raise ValueError(
                "expected_finding_count cannot exceed max_events."
            )

        return self


class TriageEvalResult(BaseModel):
    case_id: str
    status: Literal[
        "completed",
        "no_alerts",
        "incomplete_data",
        "refused",
        "invalid",
    ]
    schema_valid: bool
    status_correct: bool | None
    finding_count_correct: bool | None
    expected_findings_present: bool | None
    evidence_valid: bool
    run_consistent: bool
    max_events_respected: bool
    status_semantics_valid: bool
    passed: bool
    failure_reasons: list[str]


class TriageEvalReport(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    schema_validity_rate: float
    evidence_validity_rate: float
    run_consistency_rate: float
    max_events_compliance_rate: float
    status_semantics_rate: float
    status_accuracy: float | None
    expected_findings_accuracy: float | None
    finding_count_accuracy: float | None
    pass_rate: float
    results: list[TriageEvalResult]