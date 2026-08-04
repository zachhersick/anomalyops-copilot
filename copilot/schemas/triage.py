from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from copilot.schemas.anomaly import (
    AlertEvent,
    RowAlert,
    RunSummary,
)


class TriageRequest(BaseModel):
    run_id: Annotated[Optional[int], Field(gt=0)] = None
    max_events: int = Field(5, gt=0, le=20)


class TriageEvidence(BaseModel):
    evidence_id: str
    event: AlertEvent
    alerts: list[RowAlert]


class TriageFinding(BaseModel):
    finding_id: str
    severity: str
    machine_id: int
    sensor: str
    anomaly_type: str | None
    summary: str
    evidence_ids: list[str]


class TriageFindingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal[
        "critical",
        "warning",
        "unknown",
    ]
    machine_id: int = Field(gt=0)
    sensor: str = Field(min_length=1)
    anomaly_type: str | None
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class TriageReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "completed",
        "no_alerts",
        "incomplete_data",
        "refused",
    ]
    findings: list[TriageFindingDraft]
    refusal_reason: str | None


class TriageReport(BaseModel):
    run_id: int | None
    status: Literal[
        "completed",
        "no_alerts",
        "incomplete_data",
        "refused",
    ]
    run_summary: RunSummary | None
    findings: list[TriageFinding]
    evidence: list[TriageEvidence]
    refusal_reason: str | None = None