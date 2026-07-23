from typing import Annotated, Optional, Literal
from pydantic import BaseModel, Field

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
    
    
class TriageReport(BaseModel):
    run_id: int
    status: Literal["completed", "no_alerts"]
    run_summary: RunSummary
    findings: list[TriageFinding]
    evidence: list[TriageEvidence]